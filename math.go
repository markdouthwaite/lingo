package lingo

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// An implementation of a softmax function applied to a dense matrix.
func Softmax(x *mat.Dense) *mat.Dense {
	max := mat.Max(x)
	x.Apply(func(i, j int, v float64) float64 { return v - max }, x)
	x.Apply(func(i, j int, v float64) float64 { return math.Exp(v) }, x)
	total := mat.Sum(x)
	x.Scale(1/total, x)
	return x
}

// LinearDecisionFunction computes the decision function over the provided model and feature/s.
func LinearDecisionFunction(x *mat.Dense, coef *mat.Dense, intercept *mat.VecDense) *mat.Dense {
	ar, _ := x.Dims()
	br, _ := coef.Dims()
	H := mat.NewDense(ar, br, nil)
	H.Product(x, coef.T())
	H.Apply(func(i, j int, v float64) float64 { return v + intercept.AtVec(j) }, H)
	return H
}
