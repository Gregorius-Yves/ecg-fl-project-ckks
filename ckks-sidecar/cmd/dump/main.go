// Command dump exports the synthetic data and round results to CSV so they can
// be plotted for a presentation.
//
// Nothing else in this project writes the generated data to disk -- updates are
// fabricated in memory each run and discarded. This command materializes them.
//
//	go run ./cmd/dump                      # writes to data/
//	go run ./cmd/dump -out slides -clients 5 -preset head-only
//
// Files produced:
//
//	client_updates.csv   every client's raw update, one row per parameter
//	aggregate.csv        plaintext FedAvg vs decrypted, plus per-parameter error
//	error_hist.csv       histogram of the plaintext-vs-encrypted error
//	summary.csv          one row per parameter set: timings, payload, precision
//
// Plot suggestions for slides:
//   - aggregate.csv: scatter plaintext vs decrypted. It is a straight line.
//     That IS the result -- privacy costs nothing in utility.
//   - error_hist.csv: log-scale histogram. Errors cluster around 1e-9 while the
//     weights themselves are around 5e-2, a gap of seven orders of magnitude.
//   - summary.csv: payload against precision, showing the actual trade.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/yourname/fl-ckks-ecg/internal/crypto"
	"github.com/yourname/fl-ckks-ecg/internal/fedavg"
	"github.com/yourname/fl-ckks-ecg/internal/metrics"
	"github.com/yourname/fl-ckks-ecg/internal/model"
	"github.com/yourname/fl-ckks-ecg/internal/params"
)

func main() {
	outDir := flag.String("out", "data", "output directory")
	numClients := flag.Int("clients", 5, "number of clients")
	presetName := flag.String("preset", "head-only", "model size preset")
	mainSet := flag.String("set", "FL-D1-N8192", "parameter set used for the detailed dumps")
	flag.Parse()

	preset, err := model.LookupPreset(*presetName)
	check(err)
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		check(err)
	}

	clients := model.Cohort(2024, *numClients, preset.Params, 0.05)
	reference := model.PlaintextFedAvg(clients)
	total := model.TotalSamples(clients)

	dumpClients(filepath.Join(*outDir, "client_updates.csv"), clients, total)

	// Detailed per-parameter dump for one configuration.
	set, err := params.Lookup(*mainSet)
	check(err)
	got, _ := runRound(set, clients, total, preset)
	dumpAggregate(filepath.Join(*outDir, "aggregate.csv"), reference, got)
	dumpHistogram(filepath.Join(*outDir, "error_hist.csv"), reference, got)

	// Summary across every configuration.
	dumpSummary(filepath.Join(*outDir, "summary.csv"), clients, reference, total, preset)

	fmt.Printf("wrote 4 files to %s/\n", *outDir)
	fmt.Printf("  model  : %s (%d parameters)\n", preset.Name, preset.Params)
	fmt.Printf("  cohort : %d clients, %d total samples\n", len(clients), total)
	fmt.Printf("  detail : %s\n", set.Name)
}

func runRound(set params.Set, clients []model.Client, total int, preset model.Preset) ([]float64, fedavg.RoundStats) {
	ctx, err := crypto.New(set)
	check(err)
	srv := ctx.ServerView()

	var slowest time.Duration
	var bytesTotal int
	updates := make([]*fedavg.EncryptedUpdate, 0, len(clients))
	for _, c := range clients {
		u, err := fedavg.EncryptUpdate(ctx, c, total, set.Strategy)
		check(err)
		if u.EncryptTime > slowest {
			slowest = u.EncryptTime
		}
		bytesTotal += u.Bytes
		updates = append(updates, u)
	}

	res, err := fedavg.Aggregate(srv, updates, set.Strategy)
	check(err)
	got, decTime, err := fedavg.Decrypt(ctx, res, preset.Params)
	check(err)

	stats := fedavg.RoundStats{
		NumClients:     len(clients),
		NumParams:      preset.Params,
		Slots:          ctx.Slots(),
		ChunksPerUser:  len(updates[0].Chunks),
		EncryptTime:    slowest,
		AggregateTime:  res.AggregateTime,
		DecryptTime:    decTime,
		BytesPerClient: updates[0].Bytes,
		BytesTotal:     bytesTotal,
		Expansion:      float64(updates[0].Bytes) / float64(preset.Params*8),
	}
	return got, stats
}

func dumpClients(path string, clients []model.Client, total int) {
	w, done := newCSV(path)
	defer done()

	head := []string{"param_index"}
	for _, c := range clients {
		head = append(head, fmt.Sprintf("client_%d", c.ID))
	}
	write(w, head)

	// Metadata rows first so the file is self-describing.
	meta := []string{"#samples"}
	for _, c := range clients {
		meta = append(meta, strconv.Itoa(c.Samples))
	}
	write(w, meta)

	meta = []string{"#fedavg_weight"}
	for _, c := range clients {
		meta = append(meta, fmt.Sprintf("%.6f", c.Weight(total)))
	}
	write(w, meta)

	n := len(clients[0].Update)
	for i := 0; i < n; i++ {
		row := []string{strconv.Itoa(i)}
		for _, c := range clients {
			row = append(row, fmt.Sprintf("%.10f", c.Update[i]))
		}
		write(w, row)
	}
}

func dumpAggregate(path string, want, got []float64) {
	w, done := newCSV(path)
	defer done()
	write(w, []string{"param_index", "plaintext_fedavg", "encrypted_fedavg", "abs_error", "rel_error"})
	for i := range want {
		e := math.Abs(want[i] - got[i])
		rel := 0.0
		if want[i] != 0 {
			rel = e / math.Abs(want[i])
		}
		write(w, []string{
			strconv.Itoa(i),
			fmt.Sprintf("%.12f", want[i]),
			fmt.Sprintf("%.12f", got[i]),
			fmt.Sprintf("%.6e", e),
			fmt.Sprintf("%.6e", rel),
		})
	}
}

// dumpHistogram bins absolute error by order of magnitude. On a slide this
// makes the point instantly: the whole distribution sits far below the
// magnitude of the weights being protected.
func dumpHistogram(path string, want, got []float64) {
	w, done := newCSV(path)
	defer done()

	bins := map[int]int{}
	for i := range want {
		e := math.Abs(want[i] - got[i])
		if e == 0 {
			bins[-99]++
			continue
		}
		bins[int(math.Floor(math.Log10(e)))]++
	}

	lo, hi := 99, -99
	for k := range bins {
		if k == -99 {
			continue
		}
		if k < lo {
			lo = k
		}
		if k > hi {
			hi = k
		}
	}

	write(w, []string{"log10_abs_error", "count"})
	if bins[-99] > 0 {
		write(w, []string{"exact_zero", strconv.Itoa(bins[-99])})
	}
	for k := lo; k <= hi; k++ {
		write(w, []string{strconv.Itoa(k), strconv.Itoa(bins[k])})
	}
}

func dumpSummary(path string, clients []model.Client, reference []float64, total int, preset model.Preset) {
	w, done := newCSV(path)
	defer done()
	write(w, []string{
		"config", "logN", "depth", "scale_bits", "strategy", "slots",
		"chunks_per_client", "encrypt_ms", "aggregate_ms", "decrypt_ms",
		"payload_per_client_mb", "payload_total_mb", "expansion",
		"mae", "max_abs_error", "precision_bits",
	})

	for _, set := range params.Registry {
		got, st := runRound(set, clients, total, preset)
		p := metrics.Compare(reference, got, preset.Params)
		write(w, []string{
			set.Name,
			strconv.Itoa(set.Literal.LogN),
			strconv.Itoa(set.Depth),
			strconv.Itoa(set.Literal.LogDefaultScale),
			set.Strategy.String(),
			strconv.Itoa(st.Slots),
			strconv.Itoa(st.ChunksPerUser),
			fmt.Sprintf("%.3f", msOf(st.EncryptTime)),
			fmt.Sprintf("%.3f", msOf(st.AggregateTime)),
			fmt.Sprintf("%.3f", msOf(st.DecryptTime)),
			fmt.Sprintf("%.4f", float64(st.BytesPerClient)/1e6),
			fmt.Sprintf("%.4f", float64(st.BytesTotal)/1e6),
			fmt.Sprintf("%.2f", st.Expansion),
			fmt.Sprintf("%.6e", p.MAE),
			fmt.Sprintf("%.6e", p.MaxAE),
			fmt.Sprintf("%.2f", p.Bits),
		})
	}
}

func newCSV(path string) (*csv.Writer, func()) {
	f, err := os.Create(path)
	check(err)
	w := csv.NewWriter(f)
	return w, func() {
		w.Flush()
		check(w.Error())
		check(f.Close())
	}
}

func write(w *csv.Writer, row []string) { check(w.Write(row)) }

func msOf(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
