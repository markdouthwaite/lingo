package lingo

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNewRegressorHandler(t *testing.T) {
	var response RegressorResponse

	model := setupSimpleRegressor()

	data := `{"features": [5.0, 296.0, 16.6, 395.5, 9.04]}`
	body := strings.NewReader(data)

	req, err := http.NewRequest("POST", "/predict", body)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	handler := NewRegressorHandler(model)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}

	err = json.NewDecoder(rr.Body).Decode(&response)

	if (response.Response[0] - 28.083266491089258) > ErrorThreshold {
		t.Errorf("incorrect response retrieved")
	}

	if err != nil {
		t.Errorf("failed to get correctly formatted response")
	}
}

func TestNewClassifierHandler(t *testing.T) {
	var response ClassifierResponse

	model := setupSimpleClassifier()

	data := `{"features": [6.1, 2.8, 4.7, 1.2]}`
	body := strings.NewReader(data)

	req, err := http.NewRequest("POST", "/predict", body)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	handler := NewClassifierHandler(model)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}

	err = json.NewDecoder(rr.Body).Decode(&response)

	if response.Response[0] != 1 {
		t.Errorf("incorrect response from query")
	}

	if err != nil {
		t.Errorf("failed to get correctly formatted response")
	}
}

func TestNewRegressorHandler_InvalidFeatureDimensions(t *testing.T) {

	var response ModelError

	model := setupSimpleRegressor()

	data := `{"features": [5.0, 296.0, 16.6, 395.5]}`
	body := strings.NewReader(data)

	req, err := http.NewRequest("POST", "/predict", body)

	if err != nil {
		t.Errorf("failed to initialize request")
	}

	rr := httptest.NewRecorder()
	handler := NewRegressorHandler(model)
	handler.ServeHTTP(rr, req)

	err = json.NewDecoder(rr.Body).Decode(&response)

	if response.Message != "invalid-feature-vector" {
		t.Errorf("incorrect response message recieved")
	}

}

func TestNewRegressorHandler_InvalidQueryFormat(t *testing.T) {

	var response ModelError

	model := setupSimpleRegressor()

	data := `{"features": [5.0, 296.0, 16.6, 395.5, s]}`
	body := strings.NewReader(data)

	req, err := http.NewRequest("POST", "/predict", body)

	if err != nil {
		t.Errorf("failed to initialize request")
	}

	rr := httptest.NewRecorder()
	handler := NewRegressorHandler(model)
	handler.ServeHTTP(rr, req)

	err = json.NewDecoder(rr.Body).Decode(&response)

	if response.Message != "invalid-query-format" {
		t.Errorf("incorrect response message recieved")
	}

}
