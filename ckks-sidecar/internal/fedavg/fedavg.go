// Package fedavg implements encrypted federated averaging.
//
// A model update is longer than one ciphertext can hold, so it is split into
// chunks of `slots` parameters each. Every operation then applies chunk-wise.
// A client's encrypted update is therefore a SLICE of ciphertexts, and the
// count -- ceil(P / slots) -- is what determines bandwidth.
//
// The aggregation itself is trivial: add ciphertexts, optionally multiply by a
// plaintext scalar. No ciphertext-by-ciphertext multiplication, no rotations.
// That simplicity is the whole argument for HE in federated learning: the
// operation you need is the one HE does almost for free.
package fedavg

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"

	"github.com/yourname/fl-ckks-ecg/internal/crypto"
	"github.com/yourname/fl-ckks-ecg/internal/model"
	"github.com/yourname/fl-ckks-ecg/internal/params"
)

// NumChunks is how many ciphertexts a model of n parameters requires.
func NumChunks(n, slots int) int {
	return (n + slots - 1) / slots
}

// Chunk splits a parameter vector into slot-sized pieces, zero-padding the last.
func Chunk(w []float64, slots int) [][]float64 {
	out := make([][]float64, 0, NumChunks(len(w), slots))
	for i := 0; i < len(w); i += slots {
		end := i + slots
		piece := make([]float64, slots)
		if end > len(w) {
			copy(piece, w[i:])
		} else {
			copy(piece, w[i:end])
		}
		out = append(out, piece)
	}
	return out
}

// Unchunk reassembles chunks into a vector of exactly n parameters,
// discarding the zero padding in the final chunk.
func Unchunk(chunks [][]float64, n int) []float64 {
	out := make([]float64, 0, n)
	for _, c := range chunks {
		out = append(out, c...)
	}
	return out[:n]
}

// EncryptedUpdate is one client's contribution for one round.
type EncryptedUpdate struct {
	ClientID int
	Samples  int // n_k, sent in the clear so the server can weight
	Chunks   []*rlwe.Ciphertext

	EncryptTime time.Duration
	Bytes       int // total payload this client uploads
}

// EncryptUpdate is run CLIENT-SIDE. It chunks the update, optionally applies
// the FedAvg weight before encryption, and encrypts each chunk.
//
// With params.ClientWeighted the scaling happens here, so the server only has
// to add and no multiplicative level is consumed anywhere. The trade is that
// the client must already know the cohort's total sample count N, which costs
// an extra coordination round.
func EncryptUpdate(ctx *crypto.Context, c model.Client, totalSamples int, strategy params.Strategy) (*EncryptedUpdate, error) {
	w := c.Update
	if strategy == params.ClientWeighted {
		scaled := make([]float64, len(w))
		coeff := c.Weight(totalSamples)
		for i, v := range w {
			scaled[i] = coeff * v
		}
		w = scaled
	}

	chunks := Chunk(w, ctx.Slots())
	enc := &EncryptedUpdate{ClientID: c.ID, Samples: c.Samples}

	start := time.Now()
	for _, piece := range chunks {
		ct, err := ctx.Encrypt(piece)
		if err != nil {
			return nil, fmt.Errorf("fedavg: client %d: %w", c.ID, err)
		}
		enc.Chunks = append(enc.Chunks, ct)
		enc.Bytes += ct.BinarySize()
	}
	enc.EncryptTime = time.Since(start)

	return enc, nil
}

// AggregateResult is what the server produces.
type AggregateResult struct {
	Chunks        []*rlwe.Ciphertext
	AggregateTime time.Duration
	NumClients    int
	TotalSamples  int
}

// Aggregate is run SERVER-SIDE on ciphertexts only. The server never holds a
// secret key and never learns any individual update.
//
// ServerWeighted: multiply each client's ciphertext by the plaintext scalar
// n_k/N, then sum. Consumes one level.
//
// ClientWeighted: clients already scaled, so just sum. Consumes nothing.
func Aggregate(srv *crypto.ServerView, updates []*EncryptedUpdate, strategy params.Strategy) (*AggregateResult, error) {
	if len(updates) == 0 {
		return nil, fmt.Errorf("fedavg: no updates to aggregate")
	}

	nChunks := len(updates[0].Chunks)
	for _, u := range updates {
		if len(u.Chunks) != nChunks {
			return nil, fmt.Errorf("fedavg: client %d sent %d chunks, expected %d",
				u.ClientID, len(u.Chunks), nChunks)
		}
	}

	total := 0
	for _, u := range updates {
		total += u.Samples
	}

	res := &AggregateResult{
		Chunks:       make([]*rlwe.Ciphertext, nChunks),
		NumClients:   len(updates),
		TotalSamples: total,
	}

	eval := srv.Evaluator
	start := time.Now()

	for ci := 0; ci < nChunks; ci++ {
		var acc *rlwe.Ciphertext

		for _, u := range updates {
			term := u.Chunks[ci]

			if strategy == params.ServerWeighted {
				coeff := float64(u.Samples) / float64(total)
				scaled, err := eval.MulNew(term, coeff)
				if err != nil {
					return nil, fmt.Errorf("fedavg: scaling chunk %d: %w", ci, err)
				}
				if err := eval.Rescale(scaled, scaled); err != nil {
					return nil, fmt.Errorf("fedavg: rescaling chunk %d: %w", ci, err)
				}
				term = scaled
			}

			if acc == nil {
				acc = term.CopyNew()
				continue
			}
			if err := eval.Add(acc, term, acc); err != nil {
				return nil, fmt.Errorf("fedavg: adding chunk %d: %w", ci, err)
			}
		}

		res.Chunks[ci] = acc
	}

	res.AggregateTime = time.Since(start)
	return res, nil
}

// Decrypt is run by the KEY HOLDER. In the single-key design that is whoever
// holds sk; under threshold CKKS it becomes a quorum protocol, and this is the
// only function that changes.
func Decrypt(ctx *crypto.Context, res *AggregateResult, numParams int) ([]float64, time.Duration, error) {
	start := time.Now()
	chunks := make([][]float64, len(res.Chunks))
	for i, ct := range res.Chunks {
		out, err := ctx.Decrypt(ct)
		if err != nil {
			return nil, 0, fmt.Errorf("fedavg: decrypting chunk %d: %w", i, err)
		}
		chunks[i] = out
	}
	return Unchunk(chunks, numParams), time.Since(start), nil
}

// RoundStats summarizes one federated round for the results tables.
type RoundStats struct {
	NumClients    int
	NumParams     int
	Slots         int
	ChunksPerUser int

	EncryptTime   time.Duration // slowest single client, i.e. the wall-clock cost
	AggregateTime time.Duration
	DecryptTime   time.Duration

	BytesPerClient int
	BytesTotal     int
	Expansion      float64 // ciphertext bytes per byte of float64 payload
}

func (s RoundStats) String() string {
	return fmt.Sprintf(
		"clients=%d params=%d chunks/client=%d  encrypt=%v aggregate=%v decrypt=%v  "+
			"payload/client=%.2f MB total=%.2f MB expansion=%.1fx",
		s.NumClients, s.NumParams, s.ChunksPerUser,
		s.EncryptTime.Round(time.Millisecond),
		s.AggregateTime.Round(time.Millisecond),
		s.DecryptTime.Round(time.Millisecond),
		float64(s.BytesPerClient)/1e6, float64(s.BytesTotal)/1e6, s.Expansion)
}
