// Package threshold implements N-out-of-N threshold CKKS for federated
// aggregation.
//
// WHY THIS IS SIMPLE HERE. Threshold HE normally needs multi-round protocols to
// build relinearization and Galois keys. FedAvg needs neither: the server only
// adds ciphertexts and multiplies by plaintext scalars, and neither operation
// uses key switching. So the whole protocol reduces to two pieces:
//
//	1. Collective public key generation (one round)
//	2. Collective decryption (one round)
//
// WHAT CHANGES VERSUS SINGLE KEY. No party holds the collective secret key
//
//	s = s_1 + s_2 + ... + s_N
//
// Each party holds only its own s_i. Encryption uses the collective public key.
// Decryption requires every party to contribute a share; N-1 parties learn
// nothing.
//
// SMUDGING. Each decryption share carries added noise ("smudging"). It is part
// of the protocol, not an optional extra: without it a party could learn about
// other parties' key shares from the combined result.
//
// Smudging is also the same SHAPE of countermeasure as the noise flooding that
// IND-CPA^D requires. Whether a sigma chosen to hide key shares is large enough
// to also defeat key recovery from the published aggregate is NOT automatic --
// the two bounds are set for different reasons. cmd/threshold measures exactly
// that.
//
// API-VERIFICATION WARNING: the multiparty package names below are the part of
// this project most likely to need adjusting for your pinned Lattigo version.
// If it does not compile, the protocol structure is still right -- only the
// identifiers move.
package threshold

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// Party is one participant. It holds its own key share and nothing else.
type Party struct {
	ID int
	sk *rlwe.SecretKey
}

// Cohort is the set of parties plus the shared public material.
//
// Note what is NOT in this struct: a collective secret key. It does not exist
// in any single place, which is the entire point.
type Cohort struct {
	Params ckks.Parameters

	Parties      []*Party
	CollectivePK *rlwe.PublicKey

	// SmudgeSigma is the standard deviation of the noise each party adds to its
	// decryption share. This is the experimental knob.
	SmudgeSigma float64

	Encoder   *ckks.Encoder
	Encryptor *rlwe.Encryptor
	Evaluator *ckks.Evaluator

	zeroSk    *rlwe.SecretKey
	decryptor *rlwe.Decryptor
}

// NewCohort runs collective key generation for n parties.
//
// seed fixes the common reference string so runs are reproducible. In
// deployment the CRS comes from a public source every party can verify.
func NewCohort(params ckks.Parameters, n int, smudgeSigma float64, seed []byte) (*Cohort, error) {
	if n < 2 {
		return nil, fmt.Errorf("threshold: need at least 2 parties, got %d", n)
	}

	c := &Cohort{
		Params:      params,
		SmudgeSigma: smudgeSigma,
		Encoder:     ckks.NewEncoder(params),
	}

	// Step 1: every party samples its own secret key share, locally.
	kgen := rlwe.NewKeyGenerator(params)
	c.Parties = make([]*Party, n)
	for i := range c.Parties {
		c.Parties[i] = &Party{ID: i, sk: kgen.GenSecretKeyNew()}
	}

	// Step 2: collective public key generation. Every party derives a share
	// from the same public random polynomial; the aggregate is the collective
	// public key for s = sum_i s_i.
	crs, err := sampling.NewKeyedPRNG(seed)
	if err != nil {
		return nil, fmt.Errorf("threshold: CRS: %w", err)
	}

	ckg := multiparty.NewPublicKeyGenProtocol(params)
	crp := ckg.SampleCRP(crs)

	agg := ckg.AllocateShare()
	for i, p := range c.Parties {
		share := ckg.AllocateShare()
		ckg.GenShare(p.sk, crp, &share)
		if i == 0 {
			agg = share
			continue
		}
		ckg.AggregateShares(agg, share, &agg)
	}

	c.CollectivePK = rlwe.NewPublicKey(params)
	ckg.GenPublicKey(agg, crp, c.CollectivePK)

	c.Encryptor = rlwe.NewEncryptor(params, c.CollectivePK)

	// No evaluation keys: FedAvg uses only addition and plaintext multiplication.
	c.Evaluator = ckks.NewEvaluator(params, nil)

	// Collective decryption key-switches to the zero key, after which the
	// plaintext sits in c0 and a zero-key decryptor reads it out.
	c.zeroSk = rlwe.NewSecretKey(params)
	c.decryptor = rlwe.NewDecryptor(params, c.zeroSk)

	return c, nil
}

// Slots is N/2.
func (c *Cohort) Slots() int { return c.Params.MaxSlots() }

// NumParties is the cohort size.
func (c *Cohort) NumParties() int { return len(c.Parties) }

// Encrypt encodes and encrypts under the COLLECTIVE public key. Any party can
// do this; no party can undo it alone.
func (c *Cohort) Encrypt(values []float64) (*rlwe.Ciphertext, error) {
	pt := ckks.NewPlaintext(c.Params, c.Params.MaxLevel())
	if err := c.Encoder.Encode(values, pt); err != nil {
		return nil, fmt.Errorf("threshold: encode: %w", err)
	}
	ct, err := c.Encryptor.EncryptNew(pt)
	if err != nil {
		return nil, fmt.Errorf("threshold: encrypt: %w", err)
	}
	return ct, nil
}

// smudgeDistribution builds the noise each party adds to its decryption share.
//
// Sigma 0 means no smudging at all. That is NOT a valid threshold protocol --
// it is included only so the experiment has a baseline showing what smudging
// is protecting against.
func (c *Cohort) smudgeDistribution() ring.DistributionParameters {
	sigma := c.SmudgeSigma
	if sigma <= 0 {
		sigma = 1e-12 // effectively none; keeps the constructor happy
	}
	return ring.DiscreteGaussian{Sigma: sigma, Bound: 6 * sigma}
}

// switchToZero runs the collective decryption protocol.
//
// Every party generates a share using its own s_i and the smudging
// distribution. Shares are aggregated and applied, producing a ciphertext under
// the zero key. NO PARTY EVER SEES ANOTHER PARTY'S KEY SHARE, and a proper
// subset of parties cannot produce a usable result.
func (c *Cohort) switchToZero(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	return c.switchToZeroSubset(ct, nil)
}

// switchToZeroSubset runs the protocol using only the parties named in ids.
// nil means every party, which is the real protocol. A proper subset models a
// coalition trying to decrypt without full cooperation -- the security property
// the whole threshold construction exists to provide.
func (c *Cohort) switchToZeroSubset(ct *rlwe.Ciphertext, ids []int) (*rlwe.Ciphertext, error) {
	parties := c.Parties
	if ids != nil {
		parties = make([]*Party, 0, len(ids))
		for _, id := range ids {
			if id < 0 || id >= len(c.Parties) {
				return nil, fmt.Errorf("threshold: party %d out of range", id)
			}
			parties = append(parties, c.Parties[id])
		}
	}
	if len(parties) == 0 {
		return nil, fmt.Errorf("threshold: no parties contributed a share")
	}

	cks, err := multiparty.NewKeySwitchProtocol(c.Params, c.smudgeDistribution())
	if err != nil {
		return nil, fmt.Errorf("threshold: key-switch protocol: %w", err)
	}

	agg := cks.AllocateShare(ct.Level())
	for i, p := range parties {
		share := cks.AllocateShare(ct.Level())
		cks.GenShare(p.sk, c.zeroSk, ct, &share)
		if i == 0 {
			agg = share
			continue
		}
		cks.AggregateShares(agg, share, &agg)
	}

	out := ct.CopyNew()
	cks.KeySwitch(ct, agg, out)
	return out, nil
}

// CollectiveDecryptSubset decrypts using only the parties named in ids.
//
// This is NOT part of the protocol. It exists so the threshold property can be
// tested: with any proper subset the output must be meaningless. If a subset
// recovers the plaintext, the construction provides nothing.
func (c *Cohort) CollectiveDecryptSubset(ct *rlwe.Ciphertext, ids []int) ([]float64, error) {
	switched, err := c.switchToZeroSubset(ct, ids)
	if err != nil {
		return nil, err
	}
	pt := c.decryptor.DecryptNew(switched)
	out := make([]float64, c.Slots())
	if err := c.Encoder.Decode(pt, out); err != nil {
		return nil, fmt.Errorf("threshold: decode: %w", err)
	}
	return out, nil
}

// CollectiveDecrypt runs the full protocol and returns real slot values.
func (c *Cohort) CollectiveDecrypt(ct *rlwe.Ciphertext) ([]float64, error) {
	switched, err := c.switchToZero(ct)
	if err != nil {
		return nil, err
	}
	pt := c.decryptor.DecryptNew(switched)
	out := make([]float64, c.Slots())
	if err := c.Encoder.Decode(pt, out); err != nil {
		return nil, fmt.Errorf("threshold: decode: %w", err)
	}
	return out, nil
}

// CollectiveDecryptComplex returns the complex slot values.
//
// CKKS decodes to N/2 COMPLEX values. When the message is real the imaginary
// parts hold decryption noise, and that is what the IND-CPA^D attack needs.
// Reading only the real parts hides the leak without fixing it, so the attack
// experiment must use this form.
func (c *Cohort) CollectiveDecryptComplex(ct *rlwe.Ciphertext) ([]complex128, error) {
	switched, err := c.switchToZero(ct)
	if err != nil {
		return nil, err
	}
	pt := c.decryptor.DecryptNew(switched)
	out := make([]complex128, c.Slots())
	if err := c.Encoder.Decode(pt, out); err != nil {
		return nil, fmt.Errorf("threshold: decode: %w", err)
	}
	return out, nil
}

// CollectiveDecryptPlaintext runs the full protocol and returns the released
// plaintext POLYNOMIAL m + e, before decoding to slots.
//
// This is what the IND-CPA^D attack actually consumes. Decoding to slots and
// re-encoding is a lossy roundtrip -- Encode(Decode(pt)) != pt -- and because
// the attack multiplies by c1^{-1}, that rounding is amplified across the whole
// modulus and destroys key recovery even when no smudging is present. The
// adversary in the IND-CPA^D model is handed the decryption result itself, so
// the attack must read this polynomial, not a re-encoding of the slots.
func (c *Cohort) CollectiveDecryptPlaintext(ct *rlwe.Ciphertext) (*rlwe.Plaintext, error) {
	switched, err := c.switchToZero(ct)
	if err != nil {
		return nil, err
	}
	return c.decryptor.DecryptNew(switched), nil
}

// CollectiveSecretKeyForAttackDemo reconstructs s = sum_i s_i.
//
// THIS DOES NOT EXIST IN THE PROTOCOL. No party can compute it, and nothing
// outside the attack experiment may call it. It exists solely so the attack can
// check a recovered key against ground truth.
//
// Key shares are stored in NTT and Montgomery form, so a coefficient-wise sum
// in that domain is the correct representation of the sum.
func (c *Cohort) CollectiveSecretKeyForAttackDemo() *rlwe.SecretKey {
	ringQP := c.Params.RingQP()
	sum := rlwe.NewSecretKey(c.Params) // zero-initialised
	for _, p := range c.Parties {
		ringQP.Add(sum.Value, p.sk.Value, sum.Value)
	}
	return sum
}

// CollectiveKeyCoeffBound returns the largest absolute coefficient of the
// reconstructed collective key, in balanced representation, in the coefficient
// domain.
//
// This is a self-check on the reconstruction above, not a property of the
// scheme. Each share is ternary, so a correct sum of N shares has every
// coefficient in [-N, N]. A bound far larger than N means the reconstruction is
// wrong -- and an attack measured against a wrong ground truth would report
// "no recovery" for the wrong reason.
func (c *Cohort) CollectiveKeyCoeffBound() int64 {
	sk := c.CollectiveSecretKeyForAttackDemo()
	ringQ := c.Params.RingQ()

	std := ringQ.NewPoly()
	ringQ.IMForm(sk.Value.Q, std)
	coeff := ringQ.NewPoly()
	ringQ.INTT(std, coeff)

	q := ringQ.SubRings[0].Modulus
	var maxAbs int64
	for _, v := range coeff.Coeffs[0] {
		b := int64(v)
		if v > q/2 {
			b = int64(v) - int64(q)
		}
		if b < 0 {
			b = -b
		}
		if b > maxAbs {
			maxAbs = b
		}
	}
	return maxAbs
}

// PartyCiphertextView is what a single participant legitimately holds after a
// round: the aggregate ciphertext (needed to compute its decryption share) and
// the released plaintext (the global model). The IND-CPA^D question is whether
// that pair is enough to recover s.
type PartyCiphertextView struct {
	Aggregate *rlwe.Ciphertext
	Released  []complex128
}
