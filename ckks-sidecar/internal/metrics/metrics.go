// Package metrics compares a plaintext reference against a decrypted result.
//
// For federated aggregation the reference is plaintext FedAvg and the result
// is the decrypted encrypted aggregate. The delta between them is the headline
// number: it should be negligible, and showing that it is negligible IS the
// result.
//
// The headline number for this project is not accuracy, it is the DELTA
// between plaintext and encrypted. These functions produce that delta.
package metrics

import (
	"fmt"
	"math"
)

// Precision summarizes the difference between want and got over n slots.
type Precision struct {
	N       int
	MAE     float64 // mean absolute error
	MaxAE   float64 // worst single slot
	RMSE    float64
	RelMAE  float64 // MAE divided by mean magnitude of want
	Bits    float64 // -log2(MAE): effective bits of precision retained
}

// Compare evaluates the first n slots of want against got.
func Compare(want, got []float64, n int) Precision {
	if n > len(want) {
		n = len(want)
	}
	if n > len(got) {
		n = len(got)
	}

	var sumAbs, sumSq, maxAE, sumMag float64
	for i := 0; i < n; i++ {
		d := math.Abs(want[i] - got[i])
		sumAbs += d
		sumSq += d * d
		sumMag += math.Abs(want[i])
		if d > maxAE {
			maxAE = d
		}
	}

	p := Precision{N: n}
	if n == 0 {
		return p
	}
	p.MAE = sumAbs / float64(n)
	p.RMSE = math.Sqrt(sumSq / float64(n))
	p.MaxAE = maxAE
	if mag := sumMag / float64(n); mag > 0 {
		p.RelMAE = p.MAE / mag
	}
	if p.MAE > 0 {
		p.Bits = -math.Log2(p.MAE)
	} else {
		p.Bits = math.Inf(1)
	}
	return p
}

func (p Precision) String() string {
	return fmt.Sprintf("MAE=%.3e  max=%.3e  rel=%.3e  precision=%.1f bits  (n=%d)",
		p.MAE, p.MaxAE, p.RelMAE, p.Bits, p.N)
}

// Header and Row render the precision numbers as a TSV table for pasting into
// the results chapter.
func Header() string {
	return "config\tn\tMAE\tmaxAE\trelMAE\tbits"
}

func (p Precision) Row(config string) string {
	return fmt.Sprintf("%s\t%d\t%.6e\t%.6e\t%.6e\t%.2f",
		config, p.N, p.MAE, p.MaxAE, p.RelMAE, p.Bits)
}
