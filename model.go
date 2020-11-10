package lingo

import "gonum.org/v1/gonum/mat"

type LinearModel interface {
	Predict([]float64) []float64
}

type LinearProbabilisticClassifier interface {
	PredictProba([]float64) []float64
}

type LinearClassifier interface {
	PredictClass([]float64) []int
}

func DecisionFunction(x *mat.Dense, theta *mat.Dense, intercept *mat.VecDense) *mat.Dense {
	ar, _ := x.Dims()
	br, _ := theta.Dims()
	H := mat.NewDense(ar, br, nil)
	H.Product(x, theta.T())
	H.Apply(func(i, j int, v float64) float64 { return v + intercept.AtVec(j) }, H)
	return H
}
