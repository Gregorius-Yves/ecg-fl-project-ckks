// Command bench measures how cost scales across model size, cohort size and
// parameter set.
//
// Bandwidth is the bottleneck in federated HE, not computation. This produces
// the numbers that make that case: payload per client per round for each model
// size, and how ring dimension trades ciphertext COUNT against ciphertext SIZE.
//
//	go run ./cmd/bench
//	go run ./cmd/bench -tsv > results/bench.tsv
//	go run ./cmd/bench -scaling          # cohort size sweep
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/yourname/fl-ckks-ecg/internal/crypto"
	"github.com/yourname/fl-ckks-ecg/internal/fedavg"
	"github.com/yourname/fl-ckks-ecg/internal/model"
	"github.com/yourname/fl-ckks-ecg/internal/params"
)

func main() {
	scaling := flag.Bool("scaling", false, "sweep cohort size instead of model size")
	tsv := flag.Bool("tsv", false, "emit TSV")
	flag.Parse()

	if *scaling {
		cohortSweep(*tsv)
		return
	}
	modelSweep(*tsv)
}

// modelSweep: for each parameter set and each model size, report chunk count
// and payload. This is the table that shows why transfer learning (encrypting
// only a classifier head) is the design everyone converges on.
func modelSweep(tsv bool) {
	if tsv {
		fmt.Println("config\tslots\tmodel\tparams\tchunks\tbytesPerClient\texpansion\t" +
			"encryptMs\tkeygenMs\tpkMB\trlkMB")
	}

	for _, set := range params.Registry {
		ctx, err := crypto.New(set)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", set.Name, err)
			os.Exit(1)
		}

		if !tsv {
			fmt.Println(set.Describe())
			fmt.Printf("  keys     : pk=%.2f MB rlk=%.2f MB  keyswitching=%v  (keygen %v)\n",
				float64(ctx.Cost.PublicKeyBytes)/1e6, float64(ctx.Cost.RelinKeyBytes)/1e6,
				set.NeedsKeySwitching(), ctx.Cost.Total())
			fmt.Printf("  %-13s %10s %8s %12s %10s %10s\n",
				"model", "params", "chunks", "payload/MB", "expansion", "encrypt")
		}

		for _, preset := range model.Presets {
			clients := model.Cohort(9, 1, preset.Params, 0.05)
			u, err := fedavg.EncryptUpdate(ctx, clients[0], clients[0].Samples, set.Strategy)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s/%s: %v\n", set.Name, preset.Name, err)
				os.Exit(1)
			}

			chunks := len(u.Chunks)
			expansion := float64(u.Bytes) / float64(preset.Params*8)

			if tsv {
				fmt.Printf("%s\t%d\t%s\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.3f\t%.3f\n",
					set.Name, ctx.Slots(), preset.Name, preset.Params, chunks, u.Bytes,
					expansion, ms(u.EncryptTime), ms(ctx.Cost.Total()),
					float64(ctx.Cost.PublicKeyBytes)/1e6, float64(ctx.Cost.RelinKeyBytes)/1e6)
			} else {
				fmt.Printf("  %-13s %10d %8d %12.2f %9.1fx %8.1f ms\n",
					preset.Name, preset.Params, chunks,
					float64(u.Bytes)/1e6, expansion, ms(u.EncryptTime))
			}
		}
		if !tsv {
			fmt.Println()
		}
	}

}

// cohortSweep: fix the model, vary how many clients participate. Server
// aggregation cost should grow linearly in clients; client cost should not
// grow at all, since each client encrypts only its own update.
func cohortSweep(tsv bool) {
	set, err := params.Lookup("FL-D1-N8192")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	preset, _ := model.LookupPreset("small-cnn")

	ctx, err := crypto.New(set)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	srv := ctx.ServerView()

	if tsv {
		fmt.Println("clients\tparams\tchunks\tencryptMs\taggregateMs\tbytesTotalMB")
	} else {
		fmt.Printf("%s\nmodel: %s (%d params)\n\n", set.Describe(), preset.Name, preset.Params)
	}

	for _, n := range []int{2, 5, 10, 20, 50} {
		clients := model.Cohort(int64(n), n, preset.Params, 0.05)
		total := model.TotalSamples(clients)

		var slowest time.Duration
		var bytesTotal int
		updates := make([]*fedavg.EncryptedUpdate, 0, n)

		for _, c := range clients {
			u, err := fedavg.EncryptUpdate(ctx, c, total, set.Strategy)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			if u.EncryptTime > slowest {
				slowest = u.EncryptTime
			}
			bytesTotal += u.Bytes
			updates = append(updates, u)
		}

		res, err := fedavg.Aggregate(srv, updates, set.Strategy)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}

		if tsv {
			fmt.Printf("%d\t%d\t%d\t%.2f\t%.2f\t%.2f\n",
				n, preset.Params, len(updates[0].Chunks),
				ms(slowest), ms(res.AggregateTime), float64(bytesTotal)/1e6)
		} else {
			fmt.Printf("clients=%-3d  slowest client encrypt=%7.1f ms  "+
				"server aggregate=%7.1f ms  total payload=%7.2f MB\n",
				n, ms(slowest), ms(res.AggregateTime), float64(bytesTotal)/1e6)
		}
	}
}

func ms(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000
}
