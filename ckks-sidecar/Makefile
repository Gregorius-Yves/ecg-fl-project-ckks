.PHONY: deps roundtrip aggregate bench scaling attack attack-flood threshold security dump all clean

deps:
	go mod tidy

roundtrip:
	go run ./cmd/roundtrip

aggregate:
	go run ./cmd/aggregate

bench:
	go run ./cmd/bench

scaling:
	go run ./cmd/bench -scaling

attack:
	go run ./cmd/attack

attack-flood:
	go run ./cmd/attack -flood 1e-5

threshold:
	go run ./cmd/threshold

security:
	go run ./cmd/security

dump:
	go run ./cmd/dump

all: results/roundtrip.tsv results/aggregate.tsv results/bench.tsv results/scaling.tsv results/threshold.tsv results/security.tsv

results/roundtrip.tsv:
	go run ./cmd/roundtrip -tsv > $@

results/aggregate.tsv:
	go run ./cmd/aggregate -tsv > $@

results/bench.tsv:
	go run ./cmd/bench -tsv > $@

results/scaling.tsv:
	go run ./cmd/bench -scaling -tsv > $@

results/threshold.tsv:
	go run ./cmd/threshold -tsv > $@

results/security.tsv:
	go run ./cmd/security -test recovery -tsv > $@

clean:
	rm -f results/*.tsv data/*.csv
