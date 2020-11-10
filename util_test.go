package lingo

import (
	"testing"
)

func TestLoad_Simple(t *testing.T) {
	x := []float64{5.0, 296.0, 16.6, 395.5, 9.04}
	path := "data/regressors/simple.h5"
	model := Load(path)
	h := model.Predict(x)

	if (h[0] - 28.082343089130383) > ErrorThreshold {
		t.Errorf("Error in newly loaded model exceeds required threshold.")
	}
}

func TestLoad_Multivariate(t *testing.T) {
	x := []float64{5.0, 296.0, 16.6, 395.5, 9.04}
	path := "data/regressors/multivariate.h5"
	model := Load(path)
	h := model.Predict(x)

	if (h[0]-28.082343089130383) > ErrorThreshold || (h[1]-28.082343089130383) > ErrorThreshold {
		t.Errorf("Error in newly loaded model exceeds required threshold.")
	}
}
