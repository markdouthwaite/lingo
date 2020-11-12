package lingo

import (
	"testing"
)

func irisX() [][]float64 {
	return [][]float64{
		{6.1, 2.8, 4.7, 1.2},
		{5.7, 3.8, 1.7, 0.3},
		{7.7, 2.6, 6.9, 2.3},
		{6.0, 2.9, 4.5, 1.5},
		{6.8, 2.8, 4.8, 1.4},
		{5.4, 3.4, 1.5, 0.4},
		{5.6, 2.9, 3.6, 1.3},
		{6.9, 3.1, 5.1, 2.3},
		{6.2, 2.2, 4.5, 1.5},
		{5.8, 2.7, 3.9, 1.2},
	}
}

func irisY() []int {
	return []int{1, 0, 2, 1, 1, 0, 1, 2, 1, 1}
}

func setupSimpleClassifier() *LinearClassifier {
	theta := []float64{
		3.9351191150167972, 9.17888350905748,
		-12.377689096030275, -5.930220522464995,
		-0.7352742652265126, -1.2526120340565905,
		1.4780271634365425, -6.168284007375243,
		-3.199844849744009, -7.926271474970329,
		10.899661932611561, 12.098504529808563,
	}
	intercept := []float64{2.15291448, 20.21934658, -22.37226107}
	return NewLinearClassifier(theta, intercept, 3)
}

func TestSimpleClassifier_Predict(t *testing.T) {
	x := irisX()
	y := irisY()

	model := setupSimpleClassifier()

	for i := 0; i < len(y); i++ {
		o, err := model.Predict(x[i])

		if err != nil {
			t.Errorf("failed to complete LinearClassifier.Predict call: %s", err)
		}

		if o[0] != y[i] {
			t.Errorf("failed to return correct univariate classifier response.")
		}
	}
}
