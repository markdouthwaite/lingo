package lingo

import (
	"encoding/json"
	"io"
	"net/http"
)

type ModelQuery struct {
	Features []float64 `json:"features"`
}

type ModelResponse struct {
	Response []float64 `json:"response"`
}

type HTTPHealthChecker interface {
	Health(http.ResponseWriter, *http.Request)
}

type HTTPPredictor interface {
	Predict(http.ResponseWriter, *http.Request)
}

type HTTPModelApplication struct {
	Model LinearModel
}

func (a *HTTPModelApplication) Health(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Header().Set("Content-Type", "application/json")
	_, err := io.WriteString(w, `{"alive": true}`)

	if err != nil {
		panic("Failed to respond.")
	}
}

func (a *HTTPModelApplication) Predict(w http.ResponseWriter, r *http.Request) {
	var query ModelQuery
	err := json.NewDecoder(r.Body).Decode(&query)

	if err != nil {
		panic("Failed to decode payload.")
	}

	output := a.Model.Predict(query.Features)

	response := ModelResponse{output}

	err = json.NewEncoder(w).Encode(&response)

	w.WriteHeader(http.StatusOK)
	w.Header().Set("Content-Type", "application/json")

	if err != nil {
		panic("Failed to encode response.")
	}
}
