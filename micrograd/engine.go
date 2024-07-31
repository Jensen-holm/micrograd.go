package microgradgo

import "math"

const (
	NIL = iota
	ADD
	MUL
	POW
	TANH
	RELU
)

type Value struct {
	Data         float64
	Children     map[*Value]bool
	Op           int
	Grad         float64
	BackwardFunc func()
}

// NewValue: for constructing Values only that are not
// a part of any computational graph yet
func NewValue(data float64) *Value {
	return &Value{
		Data:         data,
		Children:     nil,
		Op:           NIL,
		BackwardFunc: nil,
		Grad:         0,
	}
}

// opResult: for constructing values from within an operation
func opResult(data float64, children map[*Value]bool, op int) *Value {
	return &Value{
		Data:         data,
		Children:     children,
		Op:           op,
		BackwardFunc: nil,
		Grad:         0,
	}
}

// newChildren: used within operators so that I don't have to
// write a map literal for every single operation
func newChildren(v, other *Value) map[*Value]bool {
	return map[*Value]bool{
		v:     true,
		other: true,
	}
}

// --- OPERATORS --- //

func (v *Value) Add(other *Value) *Value {
	result := opResult(v.Data+other.Data, newChildren(v, other), ADD)
	result.BackwardFunc = func() {
		v.Grad += result.Grad
		other.Grad += result.Grad
	}
	return result
}

func (v *Value) Mul(other *Value) *Value {
	result := opResult(v.Data*other.Data, newChildren(v, other), MUL)
	result.BackwardFunc = func() {
		v.Grad += other.Data * result.Grad
		other.Grad += v.Grad * result.Grad
	}
	return result
}

func (v *Value) Pow(other *Value) *Value {
	result := opResult(math.Pow(v.Data, other.Data), newChildren(v, other), POW)
	result.BackwardFunc = func() {
		v.Grad += (other.Data * math.Pow(v.Data, other.Data-1)) * result.Grad
	}
	return result
}

func (v *Value) Tanh() *Value {
	t := (math.Exp(2*v.Data) - 1) / (math.Exp(2 * v.Data))
	result := opResult(t, newChildren(v, nil), TANH)
	result.BackwardFunc = func() {
		v.Grad += (1 - t*t) * result.Grad
	}
	return result
}

func (v *Value) Relu() *Value {
	result := opResult(max(v.Data, 0), newChildren(v, nil), RELU)
	result.BackwardFunc = func() {
		if result.Data > 0 {
			v.Grad += result.Grad
		}
	}
	return result
}

// --- MAIN BAKCPROP METHOD --- //
