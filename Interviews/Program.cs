using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Interviews
{
    public class GraphNode : IComparable {
        public int Value;

        public Dictionary<GraphNode, int> neighbours = new Dictionary<GraphNode, int>();

        public GraphNode(int value) {
            Value = value;
        }

        public int CompareTo(object obj)
        {
            return Value.CompareTo(((GraphNode)obj).Value);
        }
    }

    public class Dijsktra
    {
        public int Solve(GraphNode node) {
            BinaryHeap<GraphNode> nextToVisit = new BinaryHeap<GraphNode>();


            return 0;
        }
    }

    public class BinaryHeap<T> : IEnumerable<T>
        where T : IComparable
    {
        private List<T> data = new List<T>();

        public void Add(T number)
        {
            data.Add(number);
            for (int i = data.Count - 1; i >= 0; i = i / 2)
            {
                if (data[(i - 1) / 2].CompareTo(data[i]) < 0)
                    break;
            }
        }

        public T PeekHead()
        {
            return data[0];
        }

        public T Pop()
        {
            if (data.Count < 1)
            {
                throw new ArgumentOutOfRangeException();
            }

            T head = data[0];
            data[0] = data[data.Count - 1];
            data.RemoveAt(data.Count - 1);
            Heapify();

            return head;
        }

        private void Heapify() {
            if (data.Count > 1) {
                for (int i = 0; i < data.Count;)
                {
                    if (data[2 * i + 1].CompareTo(data[2 * i + 2]) > 0)
                    {
                        data[i] = data[2 * i];
                        i = 2 * i + 1;
                    }
                    else
                    {
                        data[i] = data[2 * i + 2];
                        i = 2 * i + 2;
                    }
                }
            }
        }

        public string toString() {
            return string.Join(", ", data.ToArray());
        }

        public IEnumerator<T> GetEnumerator()
        {
            return data.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return data.GetEnumerator();
        }
    }

    public class Program
    {

        public static int Solve()
        {
            return 1;
        }

        public static void Main() {
            BinaryHeap binaryHeap = new BinaryHeap();
            binaryHeap.Add(1);
            binaryHeap.Add(6);
            binaryHeap.Add(0);
            binaryHeap.Add(3);
            binaryHeap.Add(4);
            binaryHeap.Add(2);

            Console.WriteLine(binaryHeap.toString());

            Console.WriteLine("hi there");
        }
    }
}
