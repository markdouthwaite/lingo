package lingo

import (
	"gonum.org/v1/gonum/mat"
)

// NewLinearClassifier initializes a new Linear LinearClassifier model
func NewLinearClassifier(coef []float64, intercept []float64, nVars int) *LinearClassifier {
	vecIntercept := mat.NewVecDense(nVars, intercept)
	newCoef := mat.NewDense(nVars, len(coef)/nVars, coef)
	model := LinearModel{newCoef, vecIntercept}
	return &LinearClassifier{&model}
}

// LinearClassifier model
type LinearClassifier struct {
	model *LinearModel
}

// Predict the probability of classes for a single observation
func (m *LinearClassifier) PredictProba(x []float64) (p []float64, err error) {
	h, err := m.model.DecisionFunction(x)

	if err != nil {
		return nil, err
	}

	p = VecToArrayFloat64(Softmax(h).RowView(0))

	return
}

//// Predict the class for a single observation
func (m *LinearClassifier) Predict(x []float64) (c []int, err error) {
	h, err := m.model.DecisionFunction(x)

	if err != nil {
		return nil, err
	}

	i, _ := Argmax(h.RowView(0))
	c = []int{i}

	return
}

//
//// Predict the probability of classes for a single observation
//func (m *LinearClassifier) Predict(x []float64) []float64 {
//	vx := mat.NewDense(1, len(x), x)
//	H := LinearDecisionFunction(vx, m.Theta, m.Intercept)
//	n, _ := m.Theta.Dims()
//	c, _ := Argmax(H.RowView(0))
//	h := make([]float64, n)
//	h[c] = 1.0
//	return h
//}
