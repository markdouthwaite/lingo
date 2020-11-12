package lingo

import (
	"math"
	"testing"
)

func TestLoad_SimpleRegressor(t *testing.T) {
	x := []float64{0.00632, 18.0, 2.31, 0.0, 0.538}
	path := "data/regressors/simple.h5"
	_, coreModel := Load(path)

	model := &LinearRegressor{coreModel}

	h, err := model.Predict(x)

	if err != nil {
		t.Errorf("LinearRegressor.Predict call failed: got error %s", err)
	}

	if math.Abs(h[0]-26.676241484140238) > ErrorThreshold {
		t.Errorf("Error in newly loaded model exceeds required threshold.")
	}
}

func TestLoad_MultivariateRegressor(t *testing.T) {
	x := []float64{0.00632, 18.0, 2.31, 0.0, 0.538}
	path := "data/regressors/multivariate.h5"
	_, coreModel := Load(path)

	model := &LinearRegressor{coreModel}

	h, err := model.Predict(x)

	if err != nil {
		t.Errorf("LinearRegressor.Predict call failed: got error %s", err)
	}

	if math.Abs(h[0]-26.676241484140238) > ErrorThreshold || math.Abs(h[1]-26.676241484140238) > ErrorThreshold {
		t.Errorf("Error in newly loaded model exceeds required threshold.")
	}
}
