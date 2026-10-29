// Command roundtrip measures the precision cost of the encryption path alone,
// with no aggregation. Encrypt a model update, decrypt it, compare.
//
// This is the correctness gate. If the roundtrip error is not near 2^-scale,
// nothing built on top of it will be trustworthy.
//
//	go run ./cmd/roundtrip
//	go run ./cmd/roundtrip -preset small-cnn -tsv > results/roundtrip.tsv
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"

	"github.com/yourname/fl-ckks-ecg/internal/crypto"
	"github.com/yourname/fl-ckks-ecg/internal/fedavg"
	"github.com/yourname/fl-ckks-ecg/internal/metrics"
	"github.com/yourname/fl-ckks-ecg/internal/model"
	"github.com/yourname/fl-ckks-ecg/internal/params"
)

func main() {
	presetName := flag.String("preset", "head-only", "model size preset")
	tsv := flag.Bool("tsv", false, "emit TSV")
	flag.Parse()

	preset, err := model.LookupPreset(*presetName)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		fmt.Fprint(os.Stderr, "available: ")
		for _, p := range model.Presets {
			fmt.Fprintf(os.Stderr, "%s ", p.Name)
		}
		fmt.Fprintln(os.Stderr)
		os.Exit(1)
	}

	if *tsv {
		fmt.Println(metrics.Header())
	} else {
		fmt.Printf("model: %s -- %d parameters (%s)\n\n", preset.Name, preset.Params, preset.Note)
	}

	rng := rand.New(rand.NewSource(1))
	update := model.Update(rng, preset.Params, 0.05)

	for _, set := range params.Registry {
		if err := run(set, update, preset, *tsv); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", set.Name, err)
			os.Exit(1)
		}
	}

}

func run(set params.Set, update []float64, preset model.Preset, tsv bool) error {
	ctx, err := crypto.New(set)
	if err != nil {
		return err
	}

	chunks := fedavg.Chunk(update, ctx.Slots())
	got := make([][]float64, len(chunks))
	totalBytes := 0

	for i, piece := range chunks {
		ct, err := ctx.Encrypt(piece)
		if err != nil {
			return err
		}
		totalBytes += ct.BinarySize()
		out, err := ctx.Decrypt(ct)
		if err != nil {
			return err
		}
		got[i] = out
	}

	recovered := fedavg.Unchunk(got, len(update))
	p := metrics.Compare(update, recovered, len(update))

	if tsv {
		fmt.Println(p.Row(set.Name))
		return nil
	}

	fmt.Println(set.Describe())
	fmt.Printf("  chunks       : %d ciphertexts for %d parameters\n", len(chunks), len(update))
	fmt.Printf("  payload      : %.3f MB  (expansion %.1fx over float64)\n",
		float64(totalBytes)/1e6, float64(totalBytes)/float64(len(update)*8))
	fmt.Printf("  keys         : pk=%.2f MB rlk=%.2f MB  (keygen %v)\n",
		float64(ctx.Cost.PublicKeyBytes)/1e6, float64(ctx.Cost.RelinKeyBytes)/1e6,
		ctx.Cost.Total())
	fmt.Printf("  roundtrip    : %s\n", p)
	fmt.Printf("  theoretical  : 2^-%d\n\n", set.Literal.LogDefaultScale)
	return nil
}
