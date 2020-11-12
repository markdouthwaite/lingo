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

// NewModelHandler creates a handler that wraps calls to the Model.Predict method.
func NewRegressorHandler(model *LinearRegressor) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {

		var query ModelQuery
		err := json.NewDecoder(r.Body).Decode(&query)

		if err != nil {
			model := ModelError{"ClientError", "Failed to decode query."}
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(&model)
		}

		output, err := model.Predict(query.Features)

		response := RegressorResponse{output}

		err = json.NewEncoder(w).Encode(&response)
		w.WriteHeader(http.StatusOK)
		w.Header().Set("Content-Type", "application/json")

	}
}
