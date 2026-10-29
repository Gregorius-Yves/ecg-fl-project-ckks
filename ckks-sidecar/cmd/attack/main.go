// Command attack demonstrates why CKKS fails IND-CPA^D (Li & Micciancio,
// EUROCRYPT 2021) and what noise flooding fixes.
//
// A NOTE ON TERMS. You cannot "test IND-CPA". It is a proof property: no
// efficient adversary distinguishes encryptions of two chosen messages with
// non-negligible advantage. No experiment establishes it, and any empirical
// check passes trivially on broken schemes. CKKS IS IND-CPA secure under
// Ring-LWE. That is not the problem.
//
// The problem is IND-CPA^D -- IND-CPA WITH A DECRYPTION ORACLE. Federated
// aggregation releases the decrypted aggregate every round, so this is the
// notion that actually applies, and CKKS fails it.
//
// THE ATTACK. Decryption computes
//
//	c0 + c1*s = m + e
//
// Normally e hides s. But CKKS returns an approximate result BY DESIGN, so the
// released output IS m+e. The adversary chose m, so it knows m, and it can see
// the output. Therefore
//
//	c1*s = (m + e) - c0
//
// is one linear equation in s with NO unknown noise term, because the noise came
// back inside the answer. c0 and c1 are public. Invert c1 and the key falls out.
// One ciphertext, one decryption, no lattice reduction.
//
// Compare BFV: decryption rounds the noise away, the output is exactly m, and
// the equation keeps an unknown. CKKS's defining design choice -- returning the
// noise as part of the answer -- is exactly what breaks it.
//
// WHY THE ORACLE RETURNS complex128. CKKS decodes to N/2 COMPLEX values. When
// the encoded message is real, the imaginary parts hold pure decryption noise --
// which is the quantity this attack needs. Decoding into []float64 silently
// discards it and the reconstruction fails. That is not a defence: it is an
// artifact of how you read the result. A real oracle returns the approximate
// decryption, and for CKKS that is complex.
//
//	go run ./cmd/attack
//	go run ./cmd/attack -flood 1e-5
//	go run ./cmd/attack -diag
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"github.com/yourname/fl-ckks-ecg/internal/crypto"
	"github.com/yourname/fl-ckks-ecg/internal/params"
)

func main() {
	setName := flag.String("set", "FL-D1-N8192", "parameter set")
	flood := flag.Float64("flood", 0, "noise flooding magnitude added before release (0 = off)")
	diag := flag.Bool("diag", false, "print reconstruction diagnostics")
	flag.Parse()

	set, err := params.Lookup(*setName)
	check(err)
	ctx, err := crypto.New(set)
	check(err)

	fmt.Println(set.Describe())
	fmt.Printf("noise flooding: %s\n\n", floodLabel(*flood))
	fmt.Printf("slots=%d  chosen-plaintext values=%d  oracle queries=1\n\n",
		ctx.Slots(), ctx.Slots())

	// The adversary CHOOSES the plaintext. Realistic here: a malicious
	// participant in a federated round picks its own "model update".
	rng := rand.New(rand.NewSource(11))
	chosen := make([]float64, ctx.Slots())
	for i := range chosen {
		chosen[i] = rng.NormFloat64() * 0.05
	}

	ct, err := ctx.Encrypt(chosen)
	check(err)

	// THE ORACLE. Submit a ciphertext, receive the approximate decryption,
	// decoded as complex128 -- what CKKS actually produces.
	ptDirect := ctx.Decryptor.DecryptNew(ct)
	released := make([]complex128, ctx.Slots())
	check(ctx.Encoder.Decode(ptDirect, released))

	if *flood > 0 {
		for i := range released {
			released[i] += complex(rng.NormFloat64()*(*flood), rng.NormFloat64()*(*flood))
		}
	}

	vulnerable := stage1(ctx, ct, released, ptDirect, *diag)
	stage2(ctx, ct, released, vulnerable)
}

// reEncode turns the oracle's output back into a plaintext polynomial at the
// ciphertext's own level and scale. This is the adversary's reconstruction of
// c0 + c1*s.
func reEncode(ctx *crypto.Context, ct *rlwe.Ciphertext, released []complex128) (*rlwe.Plaintext, error) {
	pt := ckks.NewPlaintext(ctx.Params, ct.Level())
	pt.Scale = ct.Scale
	if err := ctx.Encoder.Encode(released, pt); err != nil {
		return nil, err
	}
	return pt, nil
}

// stage1 asks whether the released output determines c0 + c1*s.
//
// If the residual is zero, the adversary's equation has no unknown. That is the
// vulnerability stated precisely, and it needs no ring inversion -- so this is
// the reliable check and the one to put on a slide.
func stage1(ctx *crypto.Context, ct *rlwe.Ciphertext, released []complex128,
	ptDirect *rlwe.Plaintext, diag bool) bool {

	fmt.Println("STAGE 1 -- does the released output determine c0 + c1*s ?")

	ringQ := ctx.Params.RingQ().AtLevel(ct.Level())

	ptGuess, err := reEncode(ctx, ct, released)
	if err != nil {
		fmt.Fprintln(os.Stderr, "  encode:", err)
		return false
	}

	// Ground truth, using the real secret key.
	sk := ctx.SecretKeyForAttackDemo()
	c0, c1 := crypto.CiphertextParts(ct)
	truth := ringQ.NewPoly()
	ringQ.MulCoeffsMontgomery(c1, sk.Value.Q, truth)
	ringQ.Add(truth, c0, truth)

	if diag {
		d := ringQ.NewPoly()
		ringQ.Sub(ptDirect.Value, truth, d)
		mx, mn := polyStats(ringQ, d)
		fmt.Printf("  [diag] truth vs direct decryption : max=%.3e mean=%.3e\n", mx, mn)
		ringQ.Sub(ptGuess.Value, ptDirect.Value, d)
		mx, mn = polyStats(ringQ, d)
		fmt.Printf("  [diag] re-encoded vs direct       : max=%.3e mean=%.3e\n", mx, mn)
	}

	resid := ringQ.NewPoly()
	ringQ.Sub(ptGuess.Value, truth, resid)
	maxDev, meanDev := polyStats(ringQ, resid)

	fmt.Printf("  residual: max=%.3e  mean=%.3e  (as a fraction of the modulus)\n", maxDev, meanDev)

	vulnerable := maxDev < 1e-9
	fmt.Printf("  reconstruction of c0 + c1*s: %s\n", exactLabel(vulnerable))
	fmt.Println()
	return vulnerable
}

// stage2 recovers the key:  s = (m' - c0) * c1^{-1}  in R_q.
//
// Ciphertexts are in NTT form, where ring multiplication is pointwise, so
// inverting c1 is a modular inverse of each NTT coefficient via Fermat.
func stage2(ctx *crypto.Context, ct *rlwe.Ciphertext, released []complex128, expectSuccess bool) {
	fmt.Println("STAGE 2 -- recover the secret key")

	ringQ := ctx.Params.RingQ().AtLevel(ct.Level())

	ptGuess, err := reEncode(ctx, ct, released)
	if err != nil {
		fmt.Fprintln(os.Stderr, "  encode:", err)
		return
	}

	c0, c1 := crypto.CiphertextParts(ct)

	// diff = m' - c0
	diff := ringQ.NewPoly()
	ringQ.Sub(ptGuess.Value, c0, diff)

	// inv = c1^{-1}, pointwise in the NTT domain
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
				fmt.Printf("  c1 not invertible at prime %d coefficient %d\n", i, j)
				return
			}
			inv.Coeffs[i][j] = ring.ModExp(v, q-2, q)
		}
	}

	recovered := ringQ.NewPoly()
	ringQ.MForm(inv, inv)
	ringQ.MulCoeffsMontgomery(diff, inv, recovered)

	// Compare in the coefficient domain against the true key.
	//
	// Lattigo stores sk.Value.Q in NTT *and* Montgomery form. MulCoeffsMontgomery
	// divides one factor by R, so stage 1's c1*sk.Value.Q lands on c1*s correctly
	// -- but the stored key itself is s*R. Meanwhile `recovered` comes out of
	// MulCoeffsMontgomery(diff, MForm(inv)) in STANDARD form, i.e. plain s.
	//
	// Comparing the two without converting matches only where a coefficient is
	// zero (0*R = 0), which in a ternary key is about a third of them. That is
	// a comparison artifact, not a failed recovery: check the ternary rate below
	// to tell the two cases apart.
	sk := ctx.SecretKeyForAttackDemo()
	skStd := ringQ.NewPoly()
	ringQ.IMForm(sk.Value.Q, skStd)

	trueS := ringQ.NewPoly()
	ringQ.INTT(skStd, trueS)
	gotS := ringQ.NewPoly()
	ringQ.INTT(recovered, gotS)

	q0 := ringQ.SubRings[0].Modulus
	match, ternary := 0, 0
	total := len(trueS.Coeffs[0])
	for j := 0; j < total; j++ {
		if trueS.Coeffs[0][j] == gotS.Coeffs[0][j] {
			match++
		}
		// A genuine CKKS secret key is ternary, so every balanced coefficient
		// must be -1, 0 or 1. Recovered garbage will not be.
		if b := balanced(gotS.Coeffs[0][j], q0); b >= -1 && b <= 1 {
			ternary++
		}
	}

	pct := 100 * float64(match) / float64(total)
	tpct := 100 * float64(ternary) / float64(total)

	fmt.Printf("  coefficients recovered exactly: %d / %d  (%.1f%%)\n", match, total, pct)
	fmt.Printf("  recovered key is ternary in   : %.1f%% of coefficients (real key: 100%%)\n", tpct)

	fmt.Printf("  key recovery                  : %s\n", recoveryLabel(pct, tpct))
	if expectSuccess && pct <= 1 {
		fmt.Println("  inconsistent: stage 1 exact but recovery failed (check inversion)")
	}
	fmt.Println()
}

func exactLabel(exact bool) string {
	if exact {
		return "EXACT (residual zero)"
	}
	return "INEXACT"
}

func recoveryLabel(pct, tpct float64) string {
	switch {
	case pct > 99:
		return "COMPLETE"
	case tpct > 99:
		return "form mismatch (recovered key is ternary; comparison domain differs)"
	case pct > 1:
		return "PARTIAL"
	default:
		return "FAILED"
	}
}

// balanced maps a coefficient into (-q/2, q/2].
func balanced(v, q uint64) int64 {
	if v > q/2 {
		return int64(v) - int64(q)
	}
	return int64(v)
}

// polyStats returns the largest and mean coefficient deviation as a fraction of
// each prime, so numbers are comparable across levels.
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

func floodLabel(f float64) string {
	if f == 0 {
		return "OFF"
	}
	return fmt.Sprintf("ON, magnitude %.1e (applied to decoded output)", f)
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
