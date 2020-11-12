package lingo

import (
	"encoding/json"
	"net/http"
)

// ModelResponse is the default query format for model requests.
type ModelQuery struct {
	Features []float64 `json:"features"`
}

// ModelResponse is the default response format from model queries.
type RegressorResponse struct {
	Response []float64 `json:"response"`
}

// ModelResponse is the default response format from model queries.
type ClassifierResponse struct {
	Response []int `json:"response"`
}

type ModelError struct {
	Title   string `json:"title"`
	Message string `json:"message"`
}

// writeBadRequest handle's client errors.
func writeBadRequest(w http.ResponseWriter, message string) {
	model := ModelError{"client-error", message}
	w.WriteHeader(http.StatusBadRequest)
	err := json.NewEncoder(w).Encode(&model)
	if err != nil {
		panic("failed to encode bad request response")
	}
}

// NewRegressorHandler creates a handler for LinearRegressors
func NewRegressorHandler(model *LinearRegressor) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {

		var query ModelQuery
		err := json.NewDecoder(r.Body).Decode(&query)

		if err != nil {
			writeBadRequest(w, "invalid-query-format")
		}

		output, err := model.Predict(query.Features)

		w.Header().Set("Content-Type", "application/json")

		if err != nil {
			writeBadRequest(w, "invalid-feature-vector")
		} else {
			w.WriteHeader(http.StatusOK)
			response := RegressorResponse{output}
			err = json.NewEncoder(w).Encode(&response)
		}

	}
}

// NewClassifierHandler creates a handler for LinearClassifiers
func NewClassifierHandler(model *LinearClassifier) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {

		var query ModelQuery
		err := json.NewDecoder(r.Body).Decode(&query)

		if err != nil {
			writeBadRequest(w, "invalid-query-format")
		}

		output, err := model.Predict(query.Features)

		w.Header().Set("Content-Type", "application/json")

		if err != nil {
			// this is unlikely to always be
			writeBadRequest(w, "invalid-feature-vector")
		} else {
			w.WriteHeader(http.StatusOK)
			response := ClassifierResponse{output}
			err = json.NewEncoder(w).Encode(&response)
		}

	}
}
