// Package crypto wraps Lattigo's CKKS objects for federated aggregation.
//
// TRUST BOUNDARY. Context holds the secret key and models the KEY HOLDER.
// ServerView strips it and models the aggregation server, which must never
// hold a secret key.
//
// Note what is absent compared to encrypted inference: no Galois keys, no
// relinearization key in the depth-0 configuration. FedAvg performs no
// ciphertext multiplication and no rotation, so key switching is either
// unnecessary or limited to a single plaintext multiply. In the inference
// project Galois keys ran to 14-99 MB; here they are zero. That saving is a
// result worth reporting.
package crypto

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"github.com/yourname/fl-ckks-ecg/internal/params"
)

// KeyCost records generation time and size for each key.
type KeyCost struct {
	SecretKeyTime time.Duration
	PublicKeyTime time.Duration
	RelinKeyTime  time.Duration

	PublicKeyBytes int
	RelinKeyBytes  int
}

func (k KeyCost) Total() time.Duration {
	return k.SecretKeyTime + k.PublicKeyTime + k.RelinKeyTime
}

// Context is the key holder's bundle.
type Context struct {
	Set    params.Set
	Params ckks.Parameters

	sk *rlwe.SecretKey
	pk *rlwe.PublicKey

	Encoder   *ckks.Encoder
	Encryptor *rlwe.Encryptor
	Decryptor *rlwe.Decryptor
	Evaluator *ckks.Evaluator

	EvkSet *rlwe.MemEvaluationKeySet
	Cost   KeyCost
}

// New builds a context for the given parameter set.
//
// A relinearization key is generated only when the set has a P basis, i.e.
// when key switching is possible at all. The depth-0 client-weighted set needs
// none, and skipping it removes megabytes from what every participant must
// store and transfer.
func New(set params.Set) (*Context, error) {
	if err := set.Check(); err != nil {
		return nil, err
	}

	p, err := ckks.NewParametersFromLiteral(set.Literal)
	if err != nil {
		return nil, fmt.Errorf("crypto: building parameters: %w", err)
	}

	c := &Context{Set: set, Params: p}
	kgen := rlwe.NewKeyGenerator(p)

	t := time.Now()
	c.sk = kgen.GenSecretKeyNew()
	c.Cost.SecretKeyTime = time.Since(t)

	t = time.Now()
	c.pk = kgen.GenPublicKeyNew(c.sk)
	c.Cost.PublicKeyTime = time.Since(t)
	c.Cost.PublicKeyBytes = c.pk.BinarySize()

	if set.NeedsKeySwitching() {
		t = time.Now()
		rlk := kgen.GenRelinearizationKeyNew(c.sk)
		c.Cost.RelinKeyTime = time.Since(t)
		c.Cost.RelinKeyBytes = rlk.BinarySize()
		c.EvkSet = rlwe.NewMemEvaluationKeySet(rlk)
	}

	c.Encoder = ckks.NewEncoder(p)
	c.Encryptor = rlwe.NewEncryptor(p, c.pk)
	c.Decryptor = rlwe.NewDecryptor(p, c.sk)
	c.Evaluator = ckks.NewEvaluator(p, c.EvkSet)

	return c, nil
}

// Slots is N/2, the number of parameters one ciphertext can hold.
func (c *Context) Slots() int { return c.Params.MaxSlots() }

// Encrypt encodes a full slot vector at the top level and encrypts it.
func (c *Context) Encrypt(values []float64) (*rlwe.Ciphertext, error) {
	pt := ckks.NewPlaintext(c.Params, c.Params.MaxLevel())
	if err := c.Encoder.Encode(values, pt); err != nil {
		return nil, fmt.Errorf("crypto: encode: %w", err)
	}
	ct, err := c.Encryptor.EncryptNew(pt)
	if err != nil {
		return nil, fmt.Errorf("crypto: encrypt: %w", err)
	}
	return ct, nil
}

// Decrypt decrypts and decodes back to a slot vector.
func (c *Context) Decrypt(ct *rlwe.Ciphertext) ([]float64, error) {
	pt := c.Decryptor.DecryptNew(ct)
	out := make([]float64, c.Slots())
	if err := c.Encoder.Decode(pt, out); err != nil {
		return nil, fmt.Errorf("crypto: decode: %w", err)
	}
	return out, nil
}

// SecretKeyForAttackDemo exposes the secret key.
//
// This exists ONLY so cmd/attack can verify a recovered key against the real
// one. A real key holder never hands this out, and nothing outside the attack
// demonstration may call it.
func (c *Context) SecretKeyForAttackDemo() *rlwe.SecretKey { return c.sk }

// CiphertextParts is a readability helper for the attack demo: a CKKS
// ciphertext is the pair (c0, c1), and decryption computes c0 + c1*s.
func CiphertextParts(ct *rlwe.Ciphertext) (c0, c1 ring.Poly) {
	return ct.Value[0], ct.Value[1]
}

// ServerView is everything the aggregation server may hold. No secret key.
type ServerView struct {
	Params    ckks.Parameters
	Evaluator *ckks.Evaluator
}

func (c *Context) ServerView() *ServerView {
	return &ServerView{Params: c.Params, Evaluator: c.Evaluator}
}

// TODO(threshold): THIS IS THE SINGLE-KEY VERSION AND IT IS NOT THE DESIGN TO
// DEFEND.
//
// With one shared key pair, HE protects each client's update from the SERVER
// but not from the other CLIENTS: any participant holding sk can decrypt any
// other participant's ciphertext, and a server colluding with a single client
// breaks the scheme entirely. A number of published FL+HE papers have exactly
// this flaw, and it is the first thing a cryptography-literate examiner probes.
//
// The fix is threshold CKKS: the secret key is shared across participants so
// that no individual can decrypt, the server aggregates blindly, and decrypting
// the SUM requires a quorum. Lattigo supports this in its multiparty package.
//
// Build order: get this single-key pipeline producing correct numbers first,
// then swap key generation and decryption for the threshold protocol. The
// aggregation logic in internal/fedavg does not change at all -- only who can
// decrypt does.
//
// TODO(ind-cpa-d): Li & Micciancio (2021). Releasing decryption results leaks
// key-dependent noise. In federated aggregation the released value is the
// aggregate, which the participants see every round, so this applies here just
// as it does to inference. Wire in noise flooding and measure the precision
// cost with cmd/roundtrip.
