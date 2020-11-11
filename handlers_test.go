package lingo

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelEndpoint_RegressionQuery(t *testing.T) {
	var response ModelResponse

	model := setupSimpleRegressor()

	data := `{"features": [5.0, 296.0, 16.6, 395.5, 9.04]}`
	body := strings.NewReader(data)

	req, err := http.NewRequest("POST", "/predict", body)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	handler := NewModelHandler(model)
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
