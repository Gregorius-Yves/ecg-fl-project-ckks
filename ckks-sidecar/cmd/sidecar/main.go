package main

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"

	"github.com/yourname/fl-ckks-ecg/internal/crypto"
	"github.com/yourname/fl-ckks-ecg/internal/params"
)

type server struct {
	mu  sync.Mutex
	ctx *crypto.Context
	srv *crypto.ServerView
	set params.Set
}

type initReq struct {
	ParamSet string `json:"param_set"`
}

type initResp struct {
	Status string `json:"status"`
	Slots  int    `json:"slots"`
	Set    string `json:"param_set"`
}

type encryptReq struct {
	WeightsB64 string `json:"weights_b64"`
	NumParams  int    `json:"num_params"`
}

type encryptResp struct {
	ChunksB64  []string `json:"chunks_b64"`
	NumChunks  int      `json:"num_chunks"`
	EncryptMs  float64  `json:"encrypt_ms"`
	BytesTotal int      `json:"bytes_total"`
}

type aggregateUpdate struct {
	ChunksB64 []string `json:"chunks_b64"`
	Weight    float64  `json:"weight"`
}

type aggregateReq struct {
	Updates []aggregateUpdate `json:"updates"`
}

type aggregateResp struct {
	ChunksB64   []string `json:"chunks_b64"`
	AggregateMs float64  `json:"aggregate_ms"`
	NumClients  int      `json:"num_clients"`
}

type decryptReq struct {
	ChunksB64 []string `json:"chunks_b64"`
	NumParams int      `json:"num_params"`
}

type decryptResp struct {
	WeightsB64    string  `json:"weights_b64"`
	DecryptMs     float64 `json:"decrypt_ms"`
	PrecisionBits float64 `json:"precision_bits,omitempty"`
}

type healthResp struct {
	Status      string `json:"status"`
	Initialized bool   `json:"initialized"`
	ParamSet    string `json:"param_set,omitempty"`
}

func float64sToBytes(v []float64) []byte {
	buf := make([]byte, len(v)*8)
	for i, f := range v {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(f))
	}
	return buf
}

func bytesToFloat64s(buf []byte) []float64 {
	n := len(buf) / 8
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[i*8:]))
	}
	return out
}

func numChunks(nParams, slots int) int {
	return (nParams + slots - 1) / slots
}

func chunkVector(w []float64, slots int) [][]float64 {
	out := make([][]float64, 0, numChunks(len(w), slots))
	for i := 0; i < len(w); i += slots {
		piece := make([]float64, slots)
		end := i + slots
		if end > len(w) {
			copy(piece, w[i:])
		} else {
			copy(piece, w[i:end])
		}
		out = append(out, piece)
	}
	return out
}

func unchunkVector(chunks [][]float64, n int) []float64 {
	out := make([]float64, 0, n)
	for _, c := range chunks {
		out = append(out, c...)
	}
	return out[:n]
}

func serializeCT(ct *rlwe.Ciphertext) ([]byte, error) {
	data, err := ct.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("serialize ciphertext: %w", err)
	}
	return data, nil
}

func deserializeCT(data []byte) (*rlwe.Ciphertext, error) {
	ct := new(rlwe.Ciphertext)
	if err := ct.UnmarshalBinary(data); err != nil {
		return nil, fmt.Errorf("deserialize ciphertext: %w", err)
	}
	return ct, nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	resp := healthResp{Status: "ok", Initialized: s.ctx != nil}
	if s.ctx != nil {
		resp.ParamSet = s.set.Name
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *server) handleInit(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req initReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	set, err := params.Lookup(req.ParamSet)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	log.Printf("[init] building CKKS context for %s ...", set.Name)
	start := time.Now()
	ctx, err := crypto.New(set)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	log.Printf("[init] context ready in %v", time.Since(start).Round(time.Millisecond))

	s.ctx = ctx
	s.srv = ctx.ServerView()
	s.set = set

	writeJSON(w, http.StatusOK, initResp{
		Status: "ok",
		Slots:  ctx.Slots(),
		Set:    set.Name,
	})
}

func (s *server) handleEncrypt(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req encryptReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	rawBytes, err := base64.StdEncoding.DecodeString(req.WeightsB64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid base64: "+err.Error())
		return
	}
	weights := bytesToFloat64s(rawBytes)
	if req.NumParams > 0 && len(weights) != req.NumParams {
		writeError(w, http.StatusBadRequest, fmt.Sprintf(
			"num_params=%d but got %d float64s", req.NumParams, len(weights)))
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx == nil {
		writeError(w, http.StatusServiceUnavailable, "call /init first")
		return
	}

	slots := s.ctx.Slots()
	chunks := chunkVector(weights, slots)

	start := time.Now()
	ctChunks := make([][]byte, len(chunks))
	bytesTotal := 0
	for i, piece := range chunks {
		ct, err := s.ctx.Encrypt(piece)
		if err != nil {
			writeError(w, http.StatusInternalServerError, fmt.Sprintf("encrypt chunk %d: %v", i, err))
			return
		}
		data, err := serializeCT(ct)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		ctChunks[i] = data
		bytesTotal += len(data)
	}
	elapsed := time.Since(start)

	b64Chunks := make([]string, len(ctChunks))
	for i, data := range ctChunks {
		b64Chunks[i] = base64.StdEncoding.EncodeToString(data)
	}

	writeJSON(w, http.StatusOK, encryptResp{
		ChunksB64:  b64Chunks,
		NumChunks:  len(b64Chunks),
		EncryptMs:  float64(elapsed.Microseconds()) / 1000,
		BytesTotal: bytesTotal,
	})
}

func (s *server) handleAggregate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req aggregateReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if len(req.Updates) == 0 {
		writeError(w, http.StatusBadRequest, "no updates provided")
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx == nil {
		writeError(w, http.StatusServiceUnavailable, "call /init first")
		return
	}

	eval := s.srv.Evaluator
	nChunks := len(req.Updates[0].ChunksB64)

	type clientCTs struct {
		chunks []*rlwe.Ciphertext
		weight float64
	}
	clients := make([]clientCTs, len(req.Updates))
	for ci, u := range req.Updates {
		if len(u.ChunksB64) != nChunks {
			writeError(w, http.StatusBadRequest, fmt.Sprintf(
				"client %d has %d chunks, expected %d", ci, len(u.ChunksB64), nChunks))
			return
		}
		clients[ci].weight = u.Weight
		clients[ci].chunks = make([]*rlwe.Ciphertext, nChunks)
		for j, b64 := range u.ChunksB64 {
			raw, err := base64.StdEncoding.DecodeString(b64)
			if err != nil {
				writeError(w, http.StatusBadRequest, fmt.Sprintf(
					"client %d chunk %d: invalid base64", ci, j))
				return
			}
			ct, err := deserializeCT(raw)
			if err != nil {
				writeError(w, http.StatusBadRequest, fmt.Sprintf(
					"client %d chunk %d: %v", ci, j, err))
				return
			}
			clients[ci].chunks[j] = ct
		}
	}

	start := time.Now()
	result := make([]*rlwe.Ciphertext, nChunks)

	for ci := 0; ci < nChunks; ci++ {
		var acc *rlwe.Ciphertext
		for _, cl := range clients {
			var term *rlwe.Ciphertext

			if cl.weight == 1.0 && len(clients) == 1 {
				term = cl.chunks[ci].CopyNew()
			} else {
				scaled, err := eval.MulNew(cl.chunks[ci], cl.weight)
				if err != nil {
					writeError(w, http.StatusInternalServerError,
						fmt.Sprintf("scaling chunk %d: %v", ci, err))
					return
				}
				if err := eval.Rescale(scaled, scaled); err != nil {
					writeError(w, http.StatusInternalServerError,
						fmt.Sprintf("rescaling chunk %d: %v", ci, err))
					return
				}
				term = scaled
			}

			if acc == nil {
				acc = term
				continue
			}
			if err := eval.Add(acc, term, acc); err != nil {
				writeError(w, http.StatusInternalServerError,
					fmt.Sprintf("adding chunk %d: %v", ci, err))
				return
			}
		}
		result[ci] = acc
	}
	elapsed := time.Since(start)

	b64Chunks := make([]string, nChunks)
	for i, ct := range result {
		data, err := serializeCT(ct)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		b64Chunks[i] = base64.StdEncoding.EncodeToString(data)
	}

	writeJSON(w, http.StatusOK, aggregateResp{
		ChunksB64:   b64Chunks,
		AggregateMs: float64(elapsed.Microseconds()) / 1000,
		NumClients:  len(clients),
	})
}

func (s *server) handleDecrypt(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req decryptReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if req.NumParams <= 0 {
		writeError(w, http.StatusBadRequest, "num_params required")
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx == nil {
		writeError(w, http.StatusServiceUnavailable, "call /init first")
		return
	}

	start := time.Now()
	decChunks := make([][]float64, len(req.ChunksB64))
	for i, b64 := range req.ChunksB64 {
		raw, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("chunk %d: invalid base64", i))
			return
		}
		ct, err := deserializeCT(raw)
		if err != nil {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("chunk %d: %v", i, err))
			return
		}
		out, err := s.ctx.Decrypt(ct)
		if err != nil {
			writeError(w, http.StatusInternalServerError, fmt.Sprintf("decrypt chunk %d: %v", i, err))
			return
		}
		decChunks[i] = out
	}
	elapsed := time.Since(start)

	flat := unchunkVector(decChunks, req.NumParams)
	b64 := base64.StdEncoding.EncodeToString(float64sToBytes(flat))

	writeJSON(w, http.StatusOK, decryptResp{
		WeightsB64: b64,
		DecryptMs:  float64(elapsed.Microseconds()) / 1000,
	})
}

func main() {
	port := flag.Int("port", 9090, "listen port")
	setName := flag.String("set", "", "auto-init with this parameter set on startup")
	flag.Parse()

	s := &server{}

	if *setName != "" {
		set, err := params.Lookup(*setName)
		if err != nil {
			log.Fatalf("unknown param set %q: %v", *setName, err)
		}
		log.Printf("auto-init: building CKKS context for %s ...", set.Name)
		start := time.Now()
		ctx, err := crypto.New(set)
		if err != nil {
			log.Fatalf("auto-init failed: %v", err)
		}
		log.Printf("auto-init: ready in %v", time.Since(start).Round(time.Millisecond))
		s.ctx = ctx
		s.srv = ctx.ServerView()
		s.set = set
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/init", s.handleInit)
	mux.HandleFunc("/encrypt", s.handleEncrypt)
	mux.HandleFunc("/aggregate", s.handleAggregate)
	mux.HandleFunc("/decrypt", s.handleDecrypt)

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("CKKS sidecar listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
