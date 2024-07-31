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

func TestNewValueprev(t *testing.T) {
	val1 := NewValue(10)
	val2 := NewValue(8.0)
	result := val1.Add(val2)

	_, val1Ok := result.prev[val1]
	_, val2Ok := result.prev[val2]
	if !val2Ok || !val1Ok {
		t.Fatalf("TestNewValueprev failed ::: prev were not added properly")
	}
}

// --- TESTING BACKPROP --- //

func TestBuildTopo(t *testing.T) {
	// ((4 * 5) + 4) * 5 = 120
	a := NewValue(4)
	b := NewValue(5)
	result := a.Mul(b).Add(a).Mul(b)
	if result.data != 120 {
		t.Fatalf("TestBuildTopo failed ::: (4 * 5 + 4) * 5 should equal 120. not '%g'", result.data)
	}

	visited := map[*Value]bool{}
	topo := []*Value{}
	topo = buildTopoOrder(result, visited, topo)
	if len(topo) != 5 {
		/// the length of topo should be the number of operations performed + 1
		t.Fatalf("TestBuildTopo failed ::: length of topo is '%d' when it should be 5 ::: topo = %#v", len(topo), topo)
	}
}

func TestAddBackward(t *testing.T) {
	a := NewValue(1)
	b := NewValue(2)
	result := a.Add(b)
	result.Backward()
	if a.grad != 1.0 {
		t.Fatalf("TestBackward failed ::: gradient of a in 'a + b' expected to be 1.0, not %g", a.grad)
	}
	if b.grad != 1.0 {
		t.Fatalf("TestBackward failed ::: gradient of b in 'a + b' expected to be 1.0, not %g", b.grad)
	}
}

func TestMulBackward(t *testing.T) {
}

func TestPowBackward(t *testing.T) {

}

func TestTanhBackward(t *testing.T) {

}

func TestReluBackward(t *testing.T) {

}
