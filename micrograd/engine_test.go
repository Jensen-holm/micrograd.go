package microgradgo

import (
	"math"
	"testing"
)

func TestNewValue(t *testing.T) {
	val := NewValue(8.0)
	if val.data != 8.0 {
		t.Fatalf("TestNewValue failed ::: %g != 8.0", val.data)
	}
	if val.op != NIL {
		t.Fatalf("TestNewValue failed ::: wrong op assigned")
	}
}

func TestAdd(t *testing.T) {
	val1 := NewValue(8.0)
	val2 := NewValue(10.0)
	result := val1.Add(val2)
	exp := val1.data + val2.data
	if result.data != exp {
		t.Fatalf("TestAdd failed ::: %g + %g != %g", val1.data, val2.data, exp)
	}
	if result.op != ADD {
		t.Fatalf("TestAdd failed ::: wrong op assigned to result")
	}
}

func TestMul(t *testing.T) {
	val1 := NewValue(100.0)
	val2 := NewValue(10.0)
	result := val1.Mul(val2)
	exp := val1.data * val2.data
	if result.data != exp {
		t.Fatalf("TestMul failed ::: %g * %g != %g", val1.data, val2.data, exp)
	}
	if result.op != MUL {
		t.Fatalf("TestMul failed ::: wrong op assigned to result")
	}
}

func TestPow(t *testing.T) {
	val1 := NewValue(1000)
	val2 := NewValue(7)
	result := val1.Pow(val2)
	exp := math.Pow(val1.data, val2.data)
	if result.data != exp {
		t.Fatalf("TestPow failed ::: %g ** %g != %g", val1.data, val2.data, exp)
	}
	if result.op != POW {
		t.Fatalf("TestPow failed ::: wrong op assigned to result")
	}
}

func TestReluNeg(t *testing.T) {
	val1 := NewValue(-1)
	result := val1.Relu()
	if result.data != 0 {
		t.Fatalf("TestReluNeg failed ::: relu(%g) != 0", val1.data)
	}
	if result.op != RELU {
		t.Fatalf("TestReluNeg failed ::: wrong op assigned to result")
	}
}

func TestReluPos(t *testing.T) {
	val1 := NewValue(1)
	result := val1.Relu()
	if result.data != val1.data {
		t.Fatalf("TestReluPos failed ::: relu(%g) != %g", val1.data, val1.data)
	}
	if result.op != RELU {
		t.Fatalf("TestReluPos failed ::: wrong op assigned to result")
	}
}

func TestNewValuechildren(t *testing.T) {
	val1 := NewValue(10)
	val2 := NewValue(8.0)
	result := val1.Add(val2)

	_, val1Ok := result.children[val1]
	_, val2Ok := result.children[val2]
	if !val2Ok || !val1Ok {
		t.Fatalf("TestNewValuechildren failed ::: children were not added properly")
	}
}

func TestBuildTopo(t *testing.T) {
	a := NewValue(4)
	b := NewValue(5)
	result := a.Mul(b).Add(a).Mul(b)
	if result.data != 120 {
		t.Fatalf("TestBuildTopo failed ::: (4 * 5 + 4) * 5 should equal 120. not '%g'", result.data)
	}

	visited := map[*Value]bool{}
	topo := []*Value{}
	topo = result.buildTopoOrder(visited, topo)
	if len(topo) != 4 {
		t.Fatalf("TestBuildTopo failed ::: length of topo is '%d' when it should be 4 ::: topo = %#v", len(topo), topo)
	}
}
