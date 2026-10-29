// Command aggregate runs a complete encrypted federated averaging round and
// compares the result against plaintext FedAvg.
//
// This is the core experiment. The claim being tested is that encrypted
// aggregation reproduces plaintext aggregation to within negligible error,
// with the cost paid in bandwidth rather than accuracy.
//
//	go run ./cmd/aggregate
//	go run ./cmd/aggregate -clients 10 -preset small-cnn
//	go run ./cmd/aggregate -tsv > results/aggregate.tsv
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/yourname/fl-ckks-ecg/internal/crypto"
	"github.com/yourname/fl-ckks-ecg/internal/fedavg"
	"github.com/yourname/fl-ckks-ecg/internal/metrics"
	"github.com/yourname/fl-ckks-ecg/internal/model"
	"github.com/yourname/fl-ckks-ecg/internal/params"
)

func main() {
	numClients := flag.Int("clients", 5, "number of participating hospitals")
	presetName := flag.String("preset", "head-only", "model size preset")
	setName := flag.String("set", "", "single parameter set (default: all)")
	tsv := flag.Bool("tsv", false, "emit TSV")
	flag.Parse()

	preset, err := model.LookupPreset(*presetName)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	sets := params.Registry
	if *setName != "" {
		s, err := params.Lookup(*setName)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		sets = []params.Set{s}
	}

	// One cohort, reused across every parameter set so the comparison is fair.
	clients := model.Cohort(2024, *numClients, preset.Params, 0.05)
	reference := model.PlaintextFedAvg(clients)
	total := model.TotalSamples(clients)

	if *tsv {
		fmt.Println("config\tstrategy\tclients\tparams\tchunks\tencryptMs\taggregateMs\t" +
			"decryptMs\tbytesPerClient\tbytesTotal\texpansion\tMAE\tmaxAE\tbits")
	} else {
		fmt.Printf("model    : %s -- %d parameters (%s)\n", preset.Name, preset.Params, preset.Note)
		fmt.Printf("cohort   : %d clients, %d total samples\n", len(clients), total)
		for _, c := range clients {
			fmt.Printf("           client %d: n_k=%-5d weight=%.4f\n",
				c.ID, c.Samples, c.Weight(total))
		}
		fmt.Println()
	}

	for _, set := range sets {
		if err := runRound(set, clients, reference, preset, total, *tsv); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", set.Name, err)
			os.Exit(1)
		}
	}

}

func runRound(set params.Set, clients []model.Client, reference []float64,
	preset model.Preset, total int, tsv bool) error {

	ctx, err := crypto.New(set)
	if err != nil {
		return err
	}
	srv := ctx.ServerView()

	// CLIENT SIDE. Each hospital encrypts locally. Rounds run in parallel in
	// reality, so the wall-clock cost is the slowest client, not the sum.
	var slowest time.Duration
	var bytesTotal int
	updates := make([]*fedavg.EncryptedUpdate, 0, len(clients))

	for _, c := range clients {
		u, err := fedavg.EncryptUpdate(ctx, c, total, set.Strategy)
		if err != nil {
			return err
		}
		if u.EncryptTime > slowest {
			slowest = u.EncryptTime
		}
		bytesTotal += u.Bytes
		updates = append(updates, u)
	}

	// SERVER SIDE. Ciphertexts only, no secret key.
	res, err := fedavg.Aggregate(srv, updates, set.Strategy)
	if err != nil {
		return err
	}

	// KEY HOLDER. Under threshold CKKS this becomes a quorum protocol.
	got, decTime, err := fedavg.Decrypt(ctx, res, preset.Params)
	if err != nil {
		return err
	}

	p := metrics.Compare(reference, got, preset.Params)
	bytesPerClient := updates[0].Bytes
	expansion := float64(bytesPerClient) / float64(preset.Params*8)
	chunks := len(updates[0].Chunks)

	if tsv {
		fmt.Printf("%s\t%s\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%d\t%d\t%.2f\t%.6e\t%.6e\t%.2f\n",
			set.Name, set.Strategy, len(clients), preset.Params, chunks,
			ms(slowest), ms(res.AggregateTime), ms(decTime),
			bytesPerClient, bytesTotal, expansion,
			p.MAE, p.MaxAE, p.Bits)
		return nil
	}

	fmt.Println(set.Describe())
	fmt.Printf("  chunks/client : %d ciphertexts\n", chunks)
	fmt.Printf("  client encrypt: %.2f ms (slowest of %d, they run in parallel)\n",
		ms(slowest), len(clients))
	fmt.Printf("  server aggreg : %.2f ms\n", ms(res.AggregateTime))
	fmt.Printf("  decrypt       : %.2f ms\n", ms(decTime))
	fmt.Printf("  payload       : %.2f MB per client, %.2f MB total per round\n",
		float64(bytesPerClient)/1e6, float64(bytesTotal)/1e6)
	fmt.Printf("  expansion     : %.1fx over plaintext float64\n", expansion)
	fmt.Printf("  vs plaintext  : %s\n\n", p)
	return nil
}

func ms(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000
}
