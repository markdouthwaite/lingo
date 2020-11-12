package lingo

import (
	"math"
	"testing"
)

func setupLinearModel() *LinearModel {
	coef := []float64{0.3574458, -0.01453134, -1.2603342, 0.00891642, -0.79181815}
	intercept := []float64{55.15045304131303}
	return NewLinearModel(coef, intercept, 1)
}

func TestLinearModel_Validate(t *testing.T) {
	x := bostonX()
	model := setupLinearModel()

	err := model.Validate(x[0])

	if err != nil {
		t.Errorf("Failed to validate properly formatted input vector.")
	}

	err = model.Validate(x[0][:4])

	if err == nil {
		t.Errorf("Failed to raise an error on an invalid input vector.")
	}
}

func TestDecisionFunction(t *testing.T) {
	x := bostonX()
	model := setupLinearModel()
	h, err := model.DecisionFunction(x[0])

	if err != nil {
		t.Errorf("DecisionFunction call failed: got error %s", err)
	}

	if math.Abs(h.At(0, 0)-28.08326571531303) > ErrorThreshold {
		t.Errorf("DecisionFunction call failed: response exceeds required threshold.")
	}
}
