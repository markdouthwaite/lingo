package lingo

import (
	"gonum.org/v1/gonum/mat"
)

// NewLinearRegressor creates a new Linear Regression model.
func NewLinearRegressor(theta []float64, intercept []float64, nVars int) *LinearRegressor {
	Intercept := mat.NewVecDense(nVars, intercept)
	Coef := mat.NewDense(nVars, len(theta)/nVars, theta)
	model := LinearModel{Coef, Intercept}
	return &LinearRegressor{&model}
}

// LinearRegressor
type LinearRegressor struct {
	model *LinearModel
}

// Predict runs inference for a single observation
func (m *LinearRegressor) Predict(x []float64) ([]float64, error) {
	h, err := m.model.DecisionFunction(x)

	if err != nil {
		return nil, err
	} else {
		return VecToArrayFloat64(h.RowView(0)), nil
	}

}
