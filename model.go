package lingo

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

type Model interface {
	Predict([]float64) ([]float64, error)
}

type ProbabilisticClassifier interface {
	PredictProba([]float64) ([]float64, error)
}

type Classifier interface {
	Predict([]float64) ([]int, error)
}

type Regressor interface {
	Predict([]float64) ([]float64, error)
}

type LinearModel struct {
	Coef      *mat.Dense
	Intercept *mat.VecDense
}

func (m *LinearModel) nTargets() int {
	r, _ := m.Coef.Dims()
	return r
}

func (m *LinearModel) nFeatures() int {
	_, c := m.Coef.Dims()
	return c
}

func (m *LinearModel) Validate(x []float64) error {

	n := m.nFeatures()

	if len(x) != n {
		return fmt.Errorf("invalid feature vector: expected length %d, got %d", len(x), n)
	}

	return nil
}

func (m *LinearModel) DecisionFunction(x []float64) (h *mat.Dense, err error) {

	defer func() {
		if err := recover(); err != nil {
			err = fmt.Errorf("failed to execute decision function")
		}
	}()

	err = m.Validate(x)

	features := mat.NewDense(1, len(x), x)
	h = LinearDecisionFunction(features, m.Coef, m.Intercept)

	return
}

func NewLinearModel(coef []float64, intercept []float64, nVars int) *LinearModel {
	Coef := mat.NewDense(nVars, len(coef)/nVars, coef)
	Intercept := mat.NewVecDense(nVars, intercept)
	return &LinearModel{Coef, Intercept}
}
