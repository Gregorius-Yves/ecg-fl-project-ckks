// Command threshold runs federated averaging under THRESHOLD CKKS and measures
// the smudging tradeoff.
//
// THE EXPERIMENT. Under threshold, no party holds the collective secret key
// s = sum_i s_i. But every party still receives, each round:
//
//   - the aggregate CIPHERTEXT (c0, c1), needed to compute its decryption share
//   - the released PLAINTEXT m + e, the global model itself
//
// That pair is exactly what the Li-Micciancio key recovery needs. Threshold
// splits the key; it does not stop the key being reconstructed from published
// output.
//
// The countermeasure is already inside the protocol: each decryption share
// carries smudging noise. But the smudging sigma is chosen to hide KEY SHARES
// from other parties, and the flooding bound required for IND-CPA^D is set for
// a different reason. Whether one suffices for the other is not automatic.
//
// This command sweeps sigma and reports, for each value:
//
//   - whether the attack still recovers the collective secret key
//   - what the smudging costs in aggregation precision
//
// Three outcomes, all worth reporting:
//
//	attack fails at protocol-standard sigma  -> threshold's smudging suffices
//	attack succeeds at protocol-standard sigma -> extra flooding is required
//	borderline                               -> quantify where the line sits
//
//	go run ./cmd/threshold
//	go run ./cmd/threshold -parties 5 -tsv > results/threshold.tsv
//	go run ./cmd/threshold -sigma 1e9 -verbose
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"github.com/yourname/fl-ckks-ecg/internal/fedavg"
	"github.com/yourname/fl-ckks-ecg/internal/metrics"
	"github.com/yourname/fl-ckks-ecg/internal/model"
	"github.com/yourname/fl-ckks-ecg/internal/params"
	"github.com/yourname/fl-ckks-ecg/internal/threshold"
)

// sweep covers no smudging, fresh-encryption scale, and increasing flooding.
var sweep = []float64{0, 3.2, 1e3, 1e6, 1e9, 1e12, 1e15}

func main() {
	setName := flag.String("set", "FL-D1-N8192", "parameter set")
	nParties := flag.Int("parties", 5, "cohort size")
	presetName := flag.String("preset", "head-only", "model size preset")
	single := flag.Float64("sigma", -1, "run a single sigma instead of the sweep")
	tsv := flag.Bool("tsv", false, "emit TSV")
	verbose := flag.Bool("verbose", false, "print per-stage attack detail")
	flag.Parse()

	set, err := params.Lookup(*setName)
	check(err)
	check(set.Check())

	preset, err := model.LookupPreset(*presetName)
	check(err)

	p, err := ckks.NewParametersFromLiteral(set.Literal)
	check(err)

	sigmas := sweep
	if *single >= 0 {
		sigmas = []float64{*single}
	}

	// One cohort of client updates, reused across every sigma so the comparison
	// is like for like.
	clients := model.Cohort(2024, *nParties, preset.Params, 0.05)
	reference := model.PlaintextFedAvg(clients)
	total := model.TotalSamples(clients)

	if *tsv {
		fmt.Println("sigma\tparties\tparams\tmae\tprecision_bits\tresidual_max\t" +
			"coeffs_recovered\tcoeffs_total\trecovered_pct\tternary_pct\tverdict\t" +
			"keygen_ms\tdecrypt_ms")
	} else {
		fmt.Println(set.Describe())
		fmt.Printf("threshold cohort: %d parties, %d-of-%d decryption\n", *nParties, *nParties, *nParties)
		fmt.Printf("model: %s (%d parameters)\n\n", preset.Name, preset.Params)
	}

	for _, sigma := range sigmas {
		runSigma(p, set, clients, reference, total, preset, *nParties, sigma, *tsv, *verbose)
	}

	if !*tsv {
		fmt.Println("sigma 0 is not a valid threshold protocol. It is the baseline")
		fmt.Println("showing what smudging protects against.")
		fmt.Println()
		fmt.Println("The attack is defeated only once flooding destroys the released")
		fmt.Println("model (verdict FLOODED). The smallest such sigma is the minimum")
		fmt.Println("flooding for IND-CPA^D -- and the precision it costs is the price.")
	}
}

func runSigma(p ckks.Parameters, set params.Set, clients []model.Client,
	reference []float64, total int, preset model.Preset,
	nParties int, sigma float64, tsv, verbose bool) {

	tKey := time.Now()
	cohort, err := threshold.NewCohort(p, nParties, sigma, []byte("fl-ckks-ecg-crs-v1"))
	check(err)
	keygenMs := msOf(time.Since(tKey))

	// ---- one federated round under threshold ----
	slots := cohort.Slots()
	nChunks := fedavg.NumChunks(preset.Params, slots)

	// Clients encrypt under the collective public key.
	encrypted := make([][]*rlwe.Ciphertext, len(clients))
	for ci, c := range clients {
		chunks := fedavg.Chunk(c.Update, slots)
		encrypted[ci] = make([]*rlwe.Ciphertext, len(chunks))
		for j, piece := range chunks {
			ct, err := cohort.Encrypt(piece)
			check(err)
			encrypted[ci][j] = ct
		}
	}

	// Server aggregates. No key of any kind.
	aggregate := make([]*rlwe.Ciphertext, nChunks)
	for j := 0; j < nChunks; j++ {
		var acc *rlwe.Ciphertext
		for ci, c := range clients {
			coeff := c.Weight(total)
			scaled, err := cohort.Evaluator.MulNew(encrypted[ci][j], coeff)
			check(err)
			check(cohort.Evaluator.Rescale(scaled, scaled))
			if ci == 0 {
				acc = scaled
				continue
			}
			check(cohort.Evaluator.Add(acc, scaled, acc))
		}
		aggregate[j] = acc
	}

	// Collective decryption: every party contributes a smudged share.
	tDec := time.Now()
	decoded := make([][]float64, nChunks)
	for j, ct := range aggregate {
		out, err := cohort.CollectiveDecrypt(ct)
		check(err)
		decoded[j] = out
	}
	decryptMs := msOf(time.Since(tDec))

	got := fedavg.Unchunk(decoded, preset.Params)
	prec := metrics.Compare(reference, got, preset.Params)

	// ---- the attack, from one participant's legitimate view ----
	target := aggregate[0]
	ptReleased, err := cohort.CollectiveDecryptPlaintext(target)
	check(err)

	residual, recovered, totalCoeffs, ternaryPct := attack(cohort, target, ptReleased)
	recPct := 100 * float64(recovered) / float64(totalCoeffs)
	verdict := verdictOf(recPct, prec.MAE)

	if tsv {
		fmt.Printf("%.3e\t%d\t%d\t%.6e\t%.2f\t%.6e\t%d\t%d\t%.2f\t%.2f\t%s\t%.2f\t%.2f\n",
			sigma, nParties, preset.Params, prec.MAE, prec.Bits, residual,
			recovered, totalCoeffs, recPct, ternaryPct, verdict, keygenMs, decryptMs)
		return
	}

	fmt.Printf("smudging sigma = %-9.1e\n", sigma)
	fmt.Printf("  aggregation vs plaintext : MAE=%.3e  precision=%.1f bits\n", prec.MAE, prec.Bits)
	fmt.Printf("  attack residual          : %.3e\n", residual)
	fmt.Printf("  coefficients recovered   : %d / %d (%.1f%%)   ternary %.1f%%\n",
		recovered, totalCoeffs, recPct, ternaryPct)
	fmt.Printf("  verdict                  : %s\n", verdict)
	if verbose {
		fmt.Printf("  keygen %.1f ms   collective decrypt %.1f ms   chunks %d\n",
			keygenMs, decryptMs, nChunks)
	}
	fmt.Println()
}

// attack runs the Li-Micciancio recovery against the collective key, using only
// what one participant legitimately holds: the aggregate ciphertext and the
// released plaintext.
func attack(cohort *threshold.Cohort, ct *rlwe.Ciphertext, ptReleased *rlwe.Plaintext) (
	residual float64, recovered, total int, ternaryPct float64) {

	p := cohort.Params
	ringQ := p.RingQ().AtLevel(ct.Level())

	// The released plaintext polynomial m + e IS what the IND-CPA^D adversary
	// receives. Decoding to slots and re-encoding would be a lossy roundtrip
	// that, after multiplication by c1^{-1}, destroys recovery on its own.
	m := ptReleased.Value

	c0, c1 := ct.Value[0], ct.Value[1]

	// Ground truth, using the collective key the protocol never materializes.
	sk := cohort.CollectiveSecretKeyForAttackDemo()
	truth := ringQ.NewPoly()
	ringQ.MulCoeffsMontgomery(c1, sk.Value.Q, truth)
	ringQ.Add(truth, c0, truth)

	// Released noise e = (m + e) - (c0 + c1*s), in the COEFFICIENT domain. These
	// polynomials live in the NTT domain, where a small noise spreads across the
	// whole modulus; measured there the residual reads ~q/2 at every sigma and
	// means nothing. (Same correction as cmd/security.)
	dNTT := ringQ.NewPoly()
	ringQ.Sub(m, truth, dNTT)
	dCoef := ringQ.NewPoly()
	ringQ.INTT(dNTT, dCoef)
	residual, _ = polyStats(ringQ, dCoef)

	// s = (m' - c0) * c1^{-1}
	diff := ringQ.NewPoly()
	ringQ.Sub(m, c0, diff)

	inv := ringQ.NewPoly()
	// Same bound as polyStats: the ciphertext's level, not the ring's prime count.
	invLevels := len(inv.Coeffs)
	if n := len(c1.Coeffs); n < invLevels {
		invLevels = n
	}
	for i := 0; i < invLevels; i++ {
		q := ringQ.SubRings[i].Modulus
		for j, v := range c1.Coeffs[i] {
			if v == 0 {
				return residual, 0, len(c1.Coeffs[0]), 0
			}
			inv.Coeffs[i][j] = ring.ModExp(v, q-2, q)
		}
	}

	rec := ringQ.NewPoly()
	ringQ.MForm(inv, inv)
	ringQ.MulCoeffsMontgomery(diff, inv, rec)

	skStd := ringQ.NewPoly()
	ringQ.IMForm(sk.Value.Q, skStd)
	trueS := ringQ.NewPoly()
	ringQ.INTT(skStd, trueS)
	gotS := ringQ.NewPoly()
	ringQ.INTT(rec, gotS)

	q0 := ringQ.SubRings[0].Modulus
	total = len(trueS.Coeffs[0])
	ternary := 0
	for j := 0; j < total; j++ {
		if trueS.Coeffs[0][j] == gotS.Coeffs[0][j] {
			recovered++
		}
		// The collective key is a sum of N ternary shares, so its coefficients
		// lie in [-N, N] rather than {-1,0,1}.
		bound := int64(cohort.NumParties())
		if b := balanced(gotS.Coeffs[0][j], q0); b >= -bound && b <= bound {
			ternary++
		}
	}
	ternaryPct = 100 * float64(ternary) / float64(total)
	return residual, recovered, total, ternaryPct
}

// verdictOf classifies one sigma. recPct (exact collective-key coefficients) is
// a knife-edge defeated by any released noise through c1^{-1} amplification, so
// it is near-zero everywhere and is NOT the safety signal. The IND-CPA^D signal
// is whether the released aggregate still carries the model (small MAE): while
// it does, the adversary holds the (ciphertext, plaintext) pair the attack needs
// and the global model has leaked. Only flooding large enough to destroy the
// model defeats the attack. (Same framing as cmd/security.)
func verdictOf(recPct, mae float64) string {
	switch {
	case recPct > 99:
		return "KEY RECOVERED"
	case mae < 1e-3:
		return "PLAINTEXT EXPOSED"
	default:
		return "FLOODED (model destroyed)"
	}
}

func balanced(v, q uint64) int64 {
	if v > q/2 {
		return int64(v) - int64(q)
	}
	return int64(v)
}

func polyStats(r *ring.Ring, poly ring.Poly) (maxDev, meanDev float64) {
	var sum float64
	var count int
	// Ring.AtLevel keeps the FULL SubRings slice, while a polynomial carries
	// only its own level's residues. A rescaled ciphertext sits below MaxLevel,
	// so iterating SubRings and indexing Coeffs runs off the end. Bound by the
	// polynomial.
	levels := len(poly.Coeffs)
	if n := len(r.SubRings); n < levels {
		levels = n
	}
	for i := 0; i < levels; i++ {
		sub := r.SubRings[i]
		q := float64(sub.Modulus)
		half := q / 2
		for _, v := range poly.Coeffs[i] {
			d := float64(v)
			if d > half {
				d -= q
			}
			d = math.Abs(d) / q
			if d > maxDev {
				maxDev = d
			}
			sum += d
			count++
		}
	}
	if count > 0 {
		meanDev = sum / float64(count)
	}
	return
}

func msOf(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
