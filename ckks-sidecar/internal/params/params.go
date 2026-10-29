// Package params holds the CKKS parameter sets for encrypted federated
// aggregation.
//
// IMPORTANT: these are NOT the parameters you would use for encrypted
// inference. FedAvg is
//
//	w_global = sum_k (n_k / N) * w_k
//
// which is ciphertext addition plus multiplication by a PLAINTEXT scalar.
// There is no ciphertext-by-ciphertext multiplication anywhere, so the
// multiplicative depth requirement collapses to 1 -- or 0 if the clients
// pre-scale their own updates before encrypting.
//
// That inverts the usual reasoning. Depth is free here. What actually matters
// is SLOT COUNT, because a model update is a long vector of reals and the
// number of ciphertexts a client must upload is ceil(P / slots). Bandwidth is
// the bottleneck in federated HE, not latency.
//
// SECURITY BUDGET (128-bit classical, HomomorphicEncryption.org standard):
//
//	LogN = 13 -> logQP <= 218    slots =  4096
//	LogN = 14 -> logQP <= 438    slots =  8192
//	LogN = 15 -> logQP <= 881    slots = 16384
//
// logQP = sum(LogQ) + sum(LogP). The P primes count. Omitting them is the
// standard way to publish parameters that are not actually 128-bit secure.
package params

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Budget caps logQP at each ring dimension for 128-bit classical security.
var Budget = map[int]int{12: 109, 13: 218, 14: 438, 15: 881}

// Strategy selects where the FedAvg weighting happens.
type Strategy int

const (
	// ServerWeighted: clients encrypt their raw update; the server multiplies
	// each ciphertext by the plaintext scalar n_k/N and sums. Costs one level.
	// The server learns nothing extra, and clients need not know the total
	// sample count N in advance.
	ServerWeighted Strategy = iota

	// ClientWeighted: each client scales its own update by n_k/N before
	// encrypting, so the server only ever adds. Costs ZERO levels, which means
	// a shorter modulus chain and smaller ciphertexts. Requires every client to
	// know N, i.e. an extra coordination round.
	ClientWeighted
)

func (s Strategy) String() string {
	if s == ClientWeighted {
		return "client-weighted"
	}
	return "server-weighted"
}

// Set is one parameter configuration plus the reason it exists.
type Set struct {
	Name      string
	Literal   ckks.ParametersLiteral
	Depth     int
	Strategy  Strategy
	Rationale string
}

// Registry holds the configurations this project benchmarks.
//
// The comparison to draw from these: ring dimension trades ciphertext COUNT
// against ciphertext SIZE. Doubling N halves the number of ciphertexts needed
// for a given model but roughly doubles each one. Whether total payload
// improves is an empirical question -- that is what cmd/bench measures.
var Registry = []Set{
	{
		Name: "FL-D0-N8192",
		Literal: ckks.ParametersLiteral{
			LogN:            13,
			LogQ:            []int{45},
			LogDefaultScale: 35,
		},
		Depth:    0,
		Strategy: ClientWeighted,
		Rationale: "Cheapest possible. Server only adds, so one prime and no P basis " +
			"at all -- no key switching means no relinearization or Galois keys. " +
			"logQP = 45 <= 218. 4096 slots.",
	},
	{
		Name: "FL-D1-N8192",
		Literal: ckks.ParametersLiteral{
			LogN:            13,
			LogQ:            []int{55, 40},
			LogP:            []int{61},
			LogDefaultScale: 40,
		},
		Depth:    1,
		Strategy: ServerWeighted,
		Rationale: "One level for the server-side plaintext scalar multiply. " +
			"logQP = 55+40+61 = 156 <= 218. 4096 slots.",
	},
	{
		Name: "FL-D1-N16384",
		Literal: ckks.ParametersLiteral{
			LogN:            14,
			LogQ:            []int{55, 40},
			LogP:            []int{61},
			LogDefaultScale: 40,
		},
		Depth:    1,
		Strategy: ServerWeighted,
		Rationale: "Same depth, double the slots (8192). Halves the ciphertext count " +
			"for a given model. Measure whether total payload actually improves.",
	},
	{
		Name: "FL-D1-N32768",
		Literal: ckks.ParametersLiteral{
			LogN:            15,
			LogQ:            []int{55, 40},
			LogP:            []int{61},
			LogDefaultScale: 40,
		},
		Depth:    1,
		Strategy: ServerWeighted,
		Rationale: "16384 slots. Far below the 881-bit budget, so security is not the " +
			"constraint here -- per-operation cost is. Expect diminishing returns.",
	},
}

func Lookup(name string) (Set, error) {
	for _, s := range Registry {
		if s.Name == name {
			return s, nil
		}
	}
	return Set{}, fmt.Errorf("params: unknown set %q", name)
}

// LogQP is the total modulus bits, which the security budget caps.
func (s Set) LogQP() int {
	n := 0
	for _, b := range s.Literal.LogQ {
		n += b
	}
	for _, b := range s.Literal.LogP {
		n += b
	}
	return n
}

// Slots is N/2, the number of real values one ciphertext can hold.
func (s Set) Slots() int { return 1 << (s.Literal.LogN - 1) }

// NeedsKeySwitching reports whether any evaluation key is required at all.
// FedAvg uses no ciphertext multiplication and no rotation, so the answer is
// almost always false -- which removes the largest storage cost in HE.
// Encrypted inference needs relinearization and Galois keys running to tens or
// hundreds of megabytes; federated aggregation needs neither.
func (s Set) NeedsKeySwitching() bool { return len(s.Literal.LogP) > 0 }

// Check validates the set against the security budget and its declared depth.
// Run before every benchmark so a parameter edit cannot silently produce
// numbers at the wrong security level.
func (s Set) Check() error {
	budget, ok := Budget[s.Literal.LogN]
	if !ok {
		return fmt.Errorf("params %s: no published budget for LogN=%d", s.Name, s.Literal.LogN)
	}
	if got := s.LogQP(); got > budget {
		return fmt.Errorf("params %s: logQP=%d exceeds the 128-bit budget of %d for LogN=%d",
			s.Name, got, budget, s.Literal.LogN)
	}
	if want := len(s.Literal.LogQ) - 1; want != s.Depth {
		return fmt.Errorf("params %s: declared depth %d but modulus chain gives %d",
			s.Name, s.Depth, want)
	}
	if s.Strategy == ServerWeighted && s.Depth < 1 {
		return fmt.Errorf("params %s: server-weighted FedAvg needs depth >= 1", s.Name)
	}
	return nil
}

func (s Set) Describe() string {
	return fmt.Sprintf("%-13s LogN=%2d  depth=%d  scale=2^%d  logQP=%3d/%3d  slots=%5d  %s",
		s.Name, s.Literal.LogN, s.Depth, s.Literal.LogDefaultScale,
		s.LogQP(), Budget[s.Literal.LogN], s.Slots(), s.Strategy)
}
