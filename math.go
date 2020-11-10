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
