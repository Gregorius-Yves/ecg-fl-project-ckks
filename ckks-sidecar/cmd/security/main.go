// Command security runs the three security tests for threshold CKKS.
//
// TEST 1 -- THRESHOLD PROPERTY. A proper subset of parties must not be able to
// decrypt. Every subset from 1 party up to N-1 is tried against a ciphertext
// whose plaintext is known. If any subset recovers it, the construction gives
// nothing over a single shared key.
//
// TEST 2 -- COLLUSION. The largest coalition short of everyone (N-1 parties)
// is the strongest realistic adversary. Reported separately because it is the
// case a reviewer asks about by name.
//
// TEST 3 -- IND-CPA^D KEY RECOVERY (Li & Micciancio, EUROCRYPT 2021). Under
// threshold, no party holds the collective key s = sum_i s_i. But each party
// still receives, every round, the aggregate CIPHERTEXT (needed to compute its
// own decryption share) and the RELEASED plaintext (the global model). That
// pair is what the attack consumes:
//
//	c0 + c1*s = m + e   ->   c1*s = (m + e) - c0   ->   s = c1^{-1}((m+e) - c0)
//
// The countermeasure already lives inside the protocol: each decryption share
// carries smudging noise. But that sigma is chosen to hide KEY SHARES from
// other parties, while the flooding bound IND-CPA^D needs is set for a
// different reason. Whether one covers the other is not automatic, so this
// sweeps sigma and reports, for each value, both whether the key falls out and
// what the smudging costs in precision.
//
//	go run ./cmd/security
//	go run ./cmd/security -parties 4 -tsv > results/security.tsv
//	go run ./cmd/security -test recovery -sigma 1e6
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"github.com/yourname/fl-ckks-ecg/internal/metrics"
	"github.com/yourname/fl-ckks-ecg/internal/params"
	"github.com/yourname/fl-ckks-ecg/internal/threshold"
)

// Sigma values swept for test 3.
//
//	0     no smudging at all -- not a valid protocol, the control
//	3.2   the standard fresh-encryption noise scale
//	rest  increasing flooding
var sigmaSweep = []float64{0, 3.2, 1e3, 1e6, 1e9, 1e12}

const crsSeed = "fl-ckks-ecg-crs-v1"

func main() {
	setName := flag.String("set", "FL-D1-N8192", "parameter set")
	nParties := flag.Int("parties", 4, "cohort size")
	which := flag.String("test", "all", "which test: all | subset | collusion | recovery")
	single := flag.Float64("sigma", -1, "run test 3 at a single sigma instead of the sweep")
	tsv := flag.Bool("tsv", false, "emit TSV for test 3")
	diag := flag.Bool("diag", false, "print reconstruction diagnostics for test 3")
	flag.Parse()

	set, err := params.Lookup(*setName)
	check(err)
	check(set.Check())

	p, err := ckks.NewParametersFromLiteral(set.Literal)
	check(err)

	if !*tsv {
		fmt.Println(set.Describe())
		fmt.Printf("cohort: %d parties, %d-of-%d decryption\n\n", *nParties, *nParties, *nParties)
	}

	run := func(name string) bool { return *which == "all" || *which == name }

	if run("subset") {
		testSubset(p, *nParties, *tsv)
	}
	if run("collusion") {
		testCollusion(p, *nParties, *tsv)
	}
	if run("recovery") {
		sigmas := sigmaSweep
		if *single >= 0 {
			sigmas = []float64{*single}
		}
		testRecovery(p, *nParties, sigmas, *tsv, *diag)
	}
}

// ---------------------------------------------------------------- test 1

func testSubset(p ckks.Parameters, n int, tsv bool) {
	if !tsv {
		fmt.Println("TEST 1 -- can a proper subset decrypt?")
	}

	cohort, err := threshold.NewCohort(p, n, 3.2, []byte(crsSeed))
	check(err)

	rng := rand.New(rand.NewSource(5))
	want := make([]float64, cohort.Slots())
	for i := range want {
		want[i] = rng.NormFloat64() * 0.05
	}
	ct, err := cohort.Encrypt(want)
	check(err)

	// Full cohort: the reference. This must succeed.
	full, err := cohort.CollectiveDecrypt(ct)
	check(err)
	fullPrec := metrics.Compare(want, full, len(want))

	if !tsv {
		fmt.Printf("  all %d parties        : MAE=%.3e  -> %s\n",
			n, fullPrec.MAE, recovered(fullPrec.MAE))
	}

	// Every proper subset size, first k parties.
	worst := math.Inf(1)
	for k := 1; k < n; k++ {
		ids := make([]int, k)
		for i := range ids {
			ids[i] = i
		}
		got, err := cohort.CollectiveDecryptSubset(ct, ids)
		check(err)
		prec := metrics.Compare(want, got, len(want))
		if prec.MAE < worst {
			worst = prec.MAE
		}
		if !tsv {
			fmt.Printf("  parties %-13s: MAE=%.3e  -> %s\n",
				idsLabel(ids), prec.MAE, recovered(prec.MAE))
		}
	}

	if !tsv {
		fmt.Printf("  verdict              : %s\n\n", subsetVerdict(fullPrec.MAE, worst))
	}
}

// ---------------------------------------------------------------- test 2

func testCollusion(p ckks.Parameters, n int, tsv bool) {
	if !tsv {
		fmt.Println("TEST 2 -- can N-1 colluding parties decrypt?")
	}
	if n < 2 {
		fmt.Println("  skipped: needs at least 2 parties")
		return
	}

	cohort, err := threshold.NewCohort(p, n, 3.2, []byte(crsSeed))
	check(err)

	rng := rand.New(rand.NewSource(6))
	want := make([]float64, cohort.Slots())
	for i := range want {
		want[i] = rng.NormFloat64() * 0.05
	}
	ct, err := cohort.Encrypt(want)
	check(err)

	// Leave out each party in turn: the coalition is everyone else.
	worst := math.Inf(1)
	for excluded := 0; excluded < n; excluded++ {
		ids := make([]int, 0, n-1)
		for i := 0; i < n; i++ {
			if i != excluded {
				ids = append(ids, i)
			}
		}
		got, err := cohort.CollectiveDecryptSubset(ct, ids)
		check(err)
		prec := metrics.Compare(want, got, len(want))
		if prec.MAE < worst {
			worst = prec.MAE
		}
		if !tsv {
			fmt.Printf("  all but party %d      : MAE=%.3e  -> %s\n",
				excluded, prec.MAE, recovered(prec.MAE))
		}
	}

	if !tsv {
		verdict := "PROTECTED (no coalition short of everyone recovers the plaintext)"
		if worst < 1e-3 {
			verdict = "BROKEN (a coalition of N-1 recovered the plaintext)"
		}
		fmt.Printf("  verdict              : %s\n\n", verdict)
	}
}

// ---------------------------------------------------------------- test 3

func testRecovery(p ckks.Parameters, n int, sigmas []float64, tsv, diag bool) {
	if tsv {
		fmt.Println("sigma\tparties\tnoise_frac\tkeybits_recovered\tkeybits_total\t" +
			"keybits_pct\tin_range_pct\tdecrypt_mae\tprecision_bits\tverdict")
	} else {
		fmt.Println("TEST 3 -- IND-CPA^D key recovery vs smudging sigma")
	}

	// Self-check the ground truth before measuring anything against it. Each
	// share is ternary, so a correct sum of N shares has every coefficient in
	// [-N, N]. If it does not, the attack would be compared against the wrong
	// key and would report "no recovery" for the wrong reason.
	probe, err := threshold.NewCohort(p, n, 3.2, []byte(crsSeed))
	check(err)
	bound := probe.CollectiveKeyCoeffBound()
	truthOK := bound <= int64(n)
	if !tsv {
		fmt.Printf("  ground truth check   : max |coeff| = %d (expected <= %d) -> %s\n",
			bound, n, map[bool]string{true: "OK", false: "WRONG, results below are meaningless"}[truthOK])
		fmt.Printf("  %-10s %12s %10s %10s %12s  %s\n",
			"sigma", "noise/q", "key bits", "in-range", "decrypt MAE", "verdict")
	}

	for _, sigma := range sigmas {
		cohort, err := threshold.NewCohort(p, n, sigma, []byte(crsSeed))
		check(err)

		rng := rand.New(rand.NewSource(11))
		chosen := make([]float64, cohort.Slots())
		for i := range chosen {
			chosen[i] = rng.NormFloat64() * 0.05
		}
		ct, err := cohort.Encrypt(chosen)
		check(err)

		// What a participant legitimately holds after a round: the released
		// plaintext polynomial m + e. Decrypt once and use the SAME draw of
		// smudging noise for both the cost measurement and the attack.
		ptReleased, err := cohort.CollectiveDecryptPlaintext(ct)
		check(err)

		// Cost side: what this sigma does to the decrypted value.
		slots := make([]complex128, cohort.Slots())
		check(cohort.Encoder.Decode(ptReleased, slots))
		real64 := make([]float64, len(slots))
		for i, v := range slots {
			real64[i] = realOf(v)
		}
		prec := metrics.Compare(chosen, real64, len(chosen))

		if diag && sigma == sigmas[0] {
			diagnose(cohort, ct, ptReleased, chosen)
		}

		residual, rec, total, inRange := attack(cohort, ct, ptReleased)
		recPct := 100 * float64(rec) / float64(total)
		v := verdictOf(recPct, prec.MAE)

		if tsv {
			fmt.Printf("%.3e\t%d\t%.6e\t%d\t%d\t%.2f\t%.2f\t%.6e\t%.2f\t%s\n",
				sigma, n, residual, rec, total, recPct, inRange, prec.MAE, prec.Bits, v)
		} else {
			fmt.Printf("  %-10.1e %12.3e %9.1f%% %9.1f%% %12.3e  %s\n",
				sigma, residual, recPct, inRange, prec.MAE, v)
		}
	}

	if !tsv {
		fmt.Println()
		if !truthOK {
			fmt.Println("  The ground-truth check failed, so every verdict above is unreliable.")
			fmt.Println("  Fix the collective-key reconstruction before reading these numbers.")
			return
		}
		fmt.Println("  sigma 0 is the control, not a valid protocol: with no smudging the")
		fmt.Println("  released plaintext is the message itself, so the attack MUST read")
		fmt.Println("  PLAINTEXT EXPOSED there. If it does not, the measurement is broken")
		fmt.Println("  rather than the scheme being safe.")
		fmt.Println()
		fmt.Println("  The 'key bits' column is exact collective-key coefficient recovery. It")
		fmt.Println("  is defeated by the baseline decryption noise alone -- c1^{-1} amplifies")
		fmt.Println("  any residual across the whole modulus -- so a low value is NOT evidence")
		fmt.Println("  of safety. The IND-CPA^D signal is whether the plaintext still leaks.")
		fmt.Println("  The reportable number is the smallest sigma whose verdict is no longer")
		fmt.Println("  PLAINTEXT EXPOSED, and the decrypt MAE it costs.")
	}
}

// attack reconstructs s from the aggregate ciphertext and the released
// plaintext: s = c1^{-1} * ((m + e) - c0).
func attack(cohort *threshold.Cohort, ct *rlwe.Ciphertext, ptReleased *rlwe.Plaintext) (
	residual float64, recovered, total int, inRangePct float64) {

	p := cohort.Params
	ringQ := p.RingQ().AtLevel(ct.Level())

	// The adversary's reconstruction of m + e IS the released plaintext
	// polynomial. Re-encoding decoded slots would be lossy; the polynomial is
	// exactly what the IND-CPA^D adversary receives.
	m := ptReleased.Value

	c0, c1 := ct.Value[0], ct.Value[1]

	// Ground truth, using the key the protocol never materializes anywhere.
	sk := cohort.CollectiveSecretKeyForAttackDemo()
	truth := ringQ.NewPoly()
	ringQ.MulCoeffsMontgomery(c1, sk.Value.Q, truth)
	ringQ.Add(truth, c0, truth)

	// The released noise e = (m + e) - (c0 + c1*s). This is the only place the
	// smudging shows up and it is exactly the quantity the IND-CPA^D flooding
	// bound governs, so it must be measured in the COEFFICIENT domain. These
	// polynomials live in the NTT domain, where a small-coefficient noise spreads
	// across the whole modulus -- measured there it would read ~q/2 at every
	// sigma and tell us nothing.
	dNTT := ringQ.NewPoly()
	ringQ.Sub(m, truth, dNTT)
	d := ringQ.NewPoly()
	ringQ.INTT(dNTT, d)
	residual, _ = polyStats(ringQ, d)

	diff := ringQ.NewPoly()
	ringQ.Sub(m, c0, diff)

	// c1^{-1}, pointwise in the NTT domain, via Fermat.
	inv := ringQ.NewPoly()
	levels := len(inv.Coeffs)
	if m := len(c1.Coeffs); m < levels {
		levels = m
	}
	for i := 0; i < levels; i++ {
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

	// Compare in the coefficient domain. Lattigo stores sk in NTT AND
	// Montgomery form; `rec` comes out in standard form, so the stored key must
	// leave Montgomery form before the comparison or only the zero coefficients
	// will match.
	skStd := ringQ.NewPoly()
	ringQ.IMForm(sk.Value.Q, skStd)
	trueS := ringQ.NewPoly()
	ringQ.INTT(skStd, trueS)
	gotS := ringQ.NewPoly()
	ringQ.INTT(rec, gotS)

	q0 := ringQ.SubRings[0].Modulus
	total = len(trueS.Coeffs[0])
	inRange := 0
	// The collective key is a sum of N ternary shares, so its coefficients lie
	// in [-N, N]. Random garbage will not. This is an independent check that a
	// high match rate is really the key and not coincidence.
	bound := int64(cohort.NumParties())
	for j := 0; j < total; j++ {
		if trueS.Coeffs[0][j] == gotS.Coeffs[0][j] {
			recovered++
		}
		if b := balanced(gotS.Coeffs[0][j], q0); b >= -bound && b <= bound {
			inRange++
		}
	}
	inRangePct = 100 * float64(inRange) / float64(total)
	return residual, recovered, total, inRangePct
}

// ---------------------------------------------------------------- helpers

// verdictOf classifies one sigma.
//
// recPct is the fraction of collective-key coefficients recovered EXACTLY. It is
// a knife-edge: s = c1^{-1}(m - c0) is exact only when the released noise is
// zero, and c1^{-1} spreads any nonzero noise across the whole modulus, so with
// even the baseline collective-decryption noise it reads ~0. A near-zero recPct
// is therefore NOT evidence the scheme is safe -- exact key bits are simply not
// the right signal.
//
// The signal that matters for IND-CPA^D is whether the released plaintext still
// carries the message. While it does (small decrypt MAE) the adversary holds the
// (ciphertext, plaintext) pair the Li-Micciancio attack consumes and the model
// itself has already leaked. Only flooding large enough to destroy the message
// defeats the attack.
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

func recovered(mae float64) string {
	if mae < 1e-3 {
		return "plaintext recovered"
	}
	return "meaningless output"
}

func subsetVerdict(fullMAE, worstSubsetMAE float64) string {
	if fullMAE >= 1e-3 {
		return "INCONCLUSIVE (the full cohort failed to decrypt; fix that first)"
	}
	if worstSubsetMAE < 1e-3 {
		return "BROKEN (a proper subset recovered the plaintext)"
	}
	return "PROTECTED (only the full cohort recovers the plaintext)"
}

func idsLabel(ids []int) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = fmt.Sprint(id)
	}
	return strings.Join(parts, ",")
}

func balanced(v, q uint64) int64 {
	if v > q/2 {
		return int64(v) - int64(q)
	}
	return int64(v)
}

func realOf(c complex128) float64 { return real(c) }

func polyStats(r *ring.Ring, poly ring.Poly) (maxDev, meanDev float64) {
	var sum float64
	var count int
	levels := len(poly.Coeffs)
	if n := len(r.SubRings); n < levels {
		levels = n
	}
	for i := 0; i < levels; i++ {
		q := float64(r.SubRings[i].Modulus)
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

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// diagnose bisects a failing test 3.
//
// If the attack reports no recovery at sigma 0, either the reconstructed ground
// truth is wrong or the adversary's re-encoding does not land on the same
// polynomial. These four lines say which.
func diagnose(cohort *threshold.Cohort, ct *rlwe.Ciphertext, ptReleased *rlwe.Plaintext, chosen []float64) {
	p := cohort.Params
	ringQ := p.RingQ().AtLevel(ct.Level())

	c0, c1 := ct.Value[0], ct.Value[1]
	sk := cohort.CollectiveSecretKeyForAttackDemo()

	truth := ringQ.NewPoly()
	ringQ.MulCoeffsMontgomery(c1, sk.Value.Q, truth)
	ringQ.Add(truth, c0, truth)

	// Does c0 + c1*s actually decode to the message? If not, the collective key
	// reconstruction is wrong and nothing measured against it means anything.
	ptTruth := ckks.NewPlaintext(p, ct.Level())
	ptTruth.Scale = ct.Scale
	ptTruth.IsNTT = ptReleased.IsNTT
	ptTruth.Value = truth
	decTruth := make([]float64, cohort.Slots())
	check(cohort.Encoder.Decode(ptTruth, decTruth))
	pT := metrics.Compare(chosen, decTruth, len(chosen))

	decGuess := make([]float64, cohort.Slots())
	check(cohort.Encoder.Decode(ptReleased, decGuess))
	pG := metrics.Compare(chosen, decGuess, len(chosen))

	// The residual m+e - (c0 + c1*s) is the smudging noise alone: tiny at sigma
	// 0, growing with sigma. It must be read in the COEFFICIENT domain -- in the
	// NTT domain these polynomials live in, a small noise spreads to ~q/2 and the
	// number is meaningless.
	residNTT := ringQ.NewPoly()
	ringQ.Sub(ptReleased.Value, truth, residNTT)
	resid := ringQ.NewPoly()
	ringQ.INTT(residNTT, resid)
	maxResid, _ := polyStats(ringQ, resid)

	fmt.Printf("  [diag] decode(c0 + c1*s) vs plaintext : MAE=%.3e\n", pT.MAE)
	fmt.Printf("  [diag] decode(released)   vs plaintext: MAE=%.3e\n", pG.MAE)
	fmt.Printf("  [diag] max |(m+e) - (c0+c1*s)| / modulus = %.3e\n", maxResid)
	fmt.Printf("  [diag] ct.IsNTT=%v ptReleased.IsNTT=%v level=%d\n",
		ct.IsNTT, ptReleased.IsNTT, ct.Level())
}
