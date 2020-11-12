package lingo

import (
	"math"
	"testing"
)

func TestLoad_Simple(t *testing.T) {
	x := []float64{0.00632, 18.0, 2.31, 0.0, 0.538}
	path := "data/regressors/simple.h5"
	model := Load(path)
	h := model.Predict(x)
	if math.Abs(h[0]-26.676241484140238) > ErrorThreshold {
		t.Errorf("Error in newly loaded model exceeds required threshold.")
	}
}

func TestLoad_Multivariate(t *testing.T) {
	x := []float64{0.00632, 18.0, 2.31, 0.0, 0.538}
	path := "data/regressors/multivariate.h5"
	model := Load(path)
	h := model.Predict(x)

	if math.Abs(h[0]-26.676241484140238) > ErrorThreshold || math.Abs(h[1]-26.676241484140238) > ErrorThreshold {
		t.Errorf("Error in newly loaded model exceeds required threshold.")
	}
}
