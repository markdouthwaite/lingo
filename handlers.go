package lingo

import (
	"encoding/json"
	"net/http"
)

type ModelQuery struct {
	Features []float64 `json:"features"`
}

type ModelResponse struct {
	Response []float64 `json:"response"`
}

func NewModelHandler(model LinearModel) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var query ModelQuery
		err := json.NewDecoder(r.Body).Decode(&query)

		if err != nil {
			panic("Failed to decode payload.")
		}

		output := model.Predict(query.Features)

		response := ModelResponse{output}

		err = json.NewEncoder(w).Encode(&response)

		w.WriteHeader(http.StatusOK)
		w.Header().Set("Content-Type", "application/json")

		if err != nil {
			panic("Failed to encode response.")
		}
	}
}
