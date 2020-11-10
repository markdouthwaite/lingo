package lingo

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelHandler_Health(t *testing.T) {

	app := HTTPModelApplication{nil}

	req, err := http.NewRequest("GET", "/health", nil)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(app.Health)
	handler.ServeHTTP(rr, req)

	// Check the status code is what we expect.
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}
	// Check the response body is what we expect.
	expected := `{"alive": true}`
	if rr.Body.String() != expected {
		t.Errorf("handler returned unexpected body: got %v want %v",
			rr.Body.String(), expected)
	}
}

func TestModelEndpoint_RegressionQuery(t *testing.T) {
	var response ModelResponse

	model := setupSimpleRegressor()
	server := HTTPModelApplication{model}

	data := `{"features": [5.0, 296.0, 16.6, 395.5, 9.04]}`
	body := strings.NewReader(data)

	req, err := http.NewRequest("POST", "/predict", body)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(server.Predict)
	handler.ServeHTTP(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}

	err = json.NewDecoder(rr.Body).Decode(&response)

	if (response.Response[0] - 28.083266491089258) > ErrorThreshold {
		t.Errorf("Incorrect response retrieved.")
	}

	if err != nil {
		t.Errorf("Failed to get correctly formatted response.")
	}

}
