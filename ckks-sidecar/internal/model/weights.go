// Package model generates synthetic model updates.
//
// This is a placeholder for real local training. In a real round each hospital
// trains on its own private ECG data and produces a weight vector; here we just
// fabricate vectors with the right length and value distribution, because the
// aggregation pipeline does not care what the numbers mean.
//
// What DOES matter and is reproduced faithfully:
//   - length (drives ciphertext count and therefore bandwidth)
//   - dynamic range (drives the precision you get at a given scale)
//
// Trained weights and gradient updates cluster near zero with a roughly normal
// distribution, which is what Update produces.
package model

import (
	"fmt"
	"math/rand"
)

// Preset is a named model size to benchmark against.
type Preset struct {
	Name   string
	Params int
	Note   string
}

// Presets span the realistic range for federated ECG work.
//
// The transfer-learning row is the important one. Encrypting 2.2M parameters
// per client per round is a bandwidth disaster; freezing a pretrained backbone
// and training only a classifier head cuts the encrypted payload by two orders
// of magnitude. That is the design most federated HE papers adopt, and the
// reason to measure both.
var Presets = []Preset{
	{"head-only", 2_565, "classifier head over a frozen backbone (5 classes)"},
	{"small-cnn", 51_205, "shallow 1D CNN trained end to end"},
	{"resnet-1d", 268_357, "deeper 1D residual network"},
	{"mobilenetv2", 2_236_682, "full backbone, end to end -- the bandwidth worst case"},
}

func LookupPreset(name string) (Preset, error) {
	for _, p := range Presets {
		if p.Name == name {
			return p, nil
		}
	}
	return Preset{}, fmt.Errorf("model: unknown preset %q", name)
}

// Update fabricates one client's model update of n parameters.
//
// scale sets the standard deviation. Real gradient updates after a few local
// epochs sit around 1e-2 to 1e-1; pass 0.05 for something representative.
func Update(rng *rand.Rand, n int, scale float64) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = rng.NormFloat64() * scale
	}
	return w
}

// Client is one participant in a federated round.
type Client struct {
	ID      int
	Samples int       // n_k, the local dataset size; drives the FedAvg weight
	Update  []float64 // w_k
}

// Cohort builds a set of clients with deliberately uneven dataset sizes,
// because real hospitals differ by an order of magnitude and equal weighting
// would hide that.
func Cohort(seed int64, numClients, numParams int, scale float64) []Client {
	rng := rand.New(rand.NewSource(seed))
	clients := make([]Client, numClients)
	for i := range clients {
		clients[i] = Client{
			ID:      i,
			Samples: 500 + rng.Intn(4500),
			Update:  Update(rng, numParams, scale),
		}
	}
	return clients
}

// TotalSamples is N = sum of n_k over the cohort.
func TotalSamples(clients []Client) int {
	total := 0
	for _, c := range clients {
		total += c.Samples
	}
	return total
}

// Weight returns this client's FedAvg coefficient n_k / N.
func (c Client) Weight(total int) float64 {
	return float64(c.Samples) / float64(total)
}

// PlaintextFedAvg computes the aggregate in the clear.
//
// This is the reference the encrypted pipeline is measured against. The
// headline result of the whole project is the DELTA between this and the
// decrypted aggregate -- not either value on its own.
func PlaintextFedAvg(clients []Client) []float64 {
	if len(clients) == 0 {
		return nil
	}
	total := TotalSamples(clients)
	out := make([]float64, len(clients[0].Update))
	for _, c := range clients {
		w := c.Weight(total)
		for i, v := range c.Update {
			out[i] += w * v
		}
	}
	return out
}
