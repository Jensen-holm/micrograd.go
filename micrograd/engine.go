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
	Data     float64
	Children map[*Value]bool
	op       int
	grad     float64
	backward func(v, other *Value)
}

// NewValue: for constructing Values only that are not
// a part of any computational graph yet
func NewValue(data float64) *Value {
	return &Value{
		Data:     data,
		Children: nil,
		op:       NIL,
		backward: nil,
		grad:     0,
	}
}

// opResult: for constructing values from within an operation
func opResult(data float64, children map[*Value]bool, op int) *Value {
	return &Value{
		Data:     data,
		Children: children,
		op:       op,
		backward: nil,
		grad:     0,
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
	v.backward = func(v, other *Value) {
		v.grad += result.grad
		other.grad += result.grad
	}
	return result
}

func (v *Value) Mul(other *Value) *Value {
	result := opResult(v.Data*other.Data, newChildren(v, other), MUL)
	v.backward = func(v, other *Value) {
		v.grad += other.Data * result.grad
		other.grad += v.grad * result.grad
	}
	return result
}

func (v *Value) Pow(other *Value) *Value {
	result := opResult(math.Pow(v.Data, other.Data), newChildren(v, other), POW)
	v.backward = func(v, other *Value) {
		v.grad += (other.Data * math.Pow(v.Data, other.Data-1)) * result.grad
	}
	return result
}

func (v *Value) Tanh() *Value {
	t := (math.Exp(2*v.Data) - 1) / (math.Exp(2 * v.Data))
	return opResult(t, newChildren(v, nil), TANH)
}

func (v *Value) Relu() *Value {
	result := opResult(v.Data, newChildren(v, nil), RELU)
	v.backward = func(v, other *Value) {
		if v.Data > 0 {
			v.grad += result.grad
		} else {
			v.grad += 0
		}
	}
	return result
}

// --- MAIN BAKCPROP METHOD --- //

func (v *Value) Backward() {

}
