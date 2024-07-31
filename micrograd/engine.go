package microgradgo

import (
	"math"
)

const (
	NIL = iota
	ADD
	MUL
	POW
	TANH
	RELU
)

type Value struct {
	data         float64
	children     map[*Value]bool
	op           int
	grad         float64
	backwardFunc func()
}

// NewValue: for constructing Values only that are not
// a part of any computational graph yet
func NewValue(data float64) *Value {
	return &Value{
		data:         data,
		children:     nil,
		op:           NIL,
		backwardFunc: nil,
		grad:         0,
	}
}

// opResult: for constructing values from within an operation
func opResult(data float64, children map[*Value]bool, op int) *Value {
	return &Value{
		data:         data,
		children:     children,
		op:           op,
		backwardFunc: nil,
		grad:         0,
	}
}

// newchildren: used within operators so that I don't have to
// write a map literal for every single operation
func newchildren(v, other *Value) map[*Value]bool {
	return map[*Value]bool{
		v:     true,
		other: true,
	}
}

// --- OPERATORS --- //

func (v *Value) Add(other *Value) *Value {
	result := opResult(v.data+other.data, newchildren(v, other), ADD)
	result.backwardFunc = func() {
		v.grad += result.grad
		other.grad += result.grad
	}
	return result
}

func (v *Value) Mul(other *Value) *Value {
	result := opResult(v.data*other.data, newchildren(v, other), MUL)
	result.backwardFunc = func() {
		v.grad += other.data * result.grad
		other.grad += v.grad * result.grad
	}
	return result
}

func (v *Value) Pow(other *Value) *Value {
	result := opResult(math.Pow(v.data, other.data), newchildren(v, other), POW)
	result.backwardFunc = func() {
		v.grad += (other.data * math.Pow(v.data, other.data-1)) * result.grad
	}
	return result
}

func (v *Value) Tanh() *Value {
	t := (math.Exp(2*v.data) - 1) / (math.Exp(2 * v.data))
	result := opResult(t, newchildren(v, nil), TANH)
	result.backwardFunc = func() {
		v.grad += (1 - t*t) * result.grad
	}
	return result
}

func (v *Value) Relu() *Value {
	result := opResult(max(v.data, 0), newchildren(v, nil), RELU)
	result.backwardFunc = func() {
		if result.data > 0 {
			v.grad += result.grad
		}
	}
	return result
}

// --- MAIN BAKCPROP METHOD & HELPERS --- //

func (v *Value) buildTopoOrder(visited map[*Value]bool, topo []*Value) []*Value {
	if _, vIsVisited := visited[v]; !vIsVisited {
		visited[v] = true
		for child := range v.children {
			child.buildTopoOrder(visited, topo)
		}
		topo = append(topo, v)
	}
	return topo
}

func (v *Value) Backward() {
	topo := make([]*Value, 0)
	visited := make(map[*Value]bool, 0)
	topo = v.buildTopoOrder(visited, topo)

	v.grad = 1
	for idx := len(topo) - 1; idx >= 0; idx-- {
		val := topo[idx]
		val.backwardFunc()
	}
}
