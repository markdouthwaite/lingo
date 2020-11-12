package lingo

import (
	"gonum.org/v1/gonum/mat"
)

type Model interface {
	Predict([]float64) []float64
}

type ProbabilisticClassifier interface {
	PredictProba([]float64) []float64
}

type Classifier interface {
	PredictClass([]float64) []int
}

// DecisionFunction computes the decision function over the provided model and observation/s.
func DecisionFunction(x *mat.Dense, theta *mat.Dense, intercept *mat.VecDense) *mat.Dense {
	ar, _ := x.Dims()
	br, _ := theta.Dims()
	H := mat.NewDense(ar, br, nil)
	H.Product(x, theta.T())
	H.Apply(func(i, j int, v float64) float64 { return v + intercept.AtVec(j) }, H)
	return H
}
