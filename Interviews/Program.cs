using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Interviews
{
    public class GraphNode : IComparable {
        public int Value { get; set; }

        public int EstimatedCost { get; set; }
        public GraphNode Previous { get; set; }

        public Dictionary<GraphNode, int> Neighbours = new Dictionary<GraphNode, int>();

        public GraphNode(int value) {
            Value = value;
        }

        public int CompareTo(object obj)
        {
            return EstimatedCost.CompareTo(((GraphNode)obj).EstimatedCost);
        }
    }

    public class Topological
    {
        public List<GraphNode> Solve(GraphNode root)
        {
            List<GraphNode> sortedNodes = new List<GraphNode>();
            HashSet<GraphNode> visited = new HashSet<GraphNode>();

            Visit(root, visited, sortedNodes);

            return sortedNodes;
        }

        private void Visit(GraphNode node, HashSet<GraphNode> visited,
                           List<GraphNode> sorted) {
            // if i wanted to detect non DAGs and throw, here i would see if
            // permanent or temporary marked
            if (visited.Contains(node)) {
                return;
            }

            visited.Add(node);

            foreach (var adjacent in node.Neighbours.Keys) {
                Visit(adjacent, visited, sorted);
            }

            sorted.Add(node);
        }
    }

    public class Dijsktra
    {
        public void AddNeighbours(GraphNode root, BinaryHeap<GraphNode> list,
                                  HashSet<GraphNode> visited)
        {
            // Add initial set of neighbours
            foreach (var neighbour in root.Neighbours.Keys)
            {
                if (!visited.Contains(neighbour))
                {
                    neighbour.EstimatedCost = int.MaxValue;
                    list.Add(neighbour);
                    visited.Add(neighbour);

                    // visit subsequent neighbours
                    AddNeighbours(neighbour, list, visited);
                }
            }
        }

        public int Solve(GraphNode node) {
            BinaryHeap<GraphNode> nextToVisit = new BinaryHeap<GraphNode>();
            node.EstimatedCost = 0;
            nextToVisit.Add(node);

            AddNeighbours(node, nextToVisit, new HashSet<GraphNode> { node });

            HashSet<GraphNode> visited = new HashSet<GraphNode>();
            while (nextToVisit.Count > 0)
            {
                GraphNode current = nextToVisit.Pop();
                visited.Add(current);

                foreach (var n in current.Neighbours) {
                    if (!visited.Contains(n.Key)) {
                        if (current.EstimatedCost + n.Value < n.Key.EstimatedCost) {
                            
                            n.Key.EstimatedCost = current.EstimatedCost + n.Value;
                            n.Key.Previous = current;
                            nextToVisit.Update(n.Key);
                        }
                    }
                }
            }

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
            for (int i = data.Count - 1; i > 0; i = (i - 1) / 2)
            {
                if (data[(i - 1) / 2].CompareTo(data[i]) < 0)
                    break;

                Swap(i, (i - 1) / 2);
            }
        }

        public T PeekHead()
        {
            return data[0];
        }

        public int Count {
            get {
                return data.Count;
            }
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

        private void Swap(int index1, int index2)
        {
            T tmp = data[index1];
            data[index1] = data[index2];
            data[index2] = tmp; 
        }

        public void Update(T node)
        {
            for (int i = 0; i < data.Count; i++) {
                if (node.CompareTo(data[i]) == 0) {
                    // heapify from here
                    for (int j = i; j > 0; ) {
                        if (data[(j - 1) / 2].CompareTo(data[i]) > 0)
                        {
                            Swap((j - 1) / 2, j);
                            j = (j - 1) / 2;
                        }
                        else return;
                    }
                }
            }
        }

        private void Heapify()
        {
            if (data.Count > 1) {
                for (int i = 0; (2 * i + 1) < data.Count;)
                {
                    T smallest = data[2 * i + 1];
                    int smallestIndex = 2 * i + 1;
                    if (2 * i + 2 < data.Count && smallest.CompareTo(data[2 * i + 2]) > 0) {
                        smallest = data[2 * i + 2];
                        smallestIndex = 2 * i + 2;
                    }

                    if (data[i].CompareTo(smallest) < 0) {
                        break;
                    }

                    Swap(i, smallestIndex);
                    i = smallestIndex;
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

        public static void Main()
        {
            BinaryHeap<int> binaryHeap = new BinaryHeap<int>();
            binaryHeap.Add(1);
            binaryHeap.Add(6);
            binaryHeap.Add(0);
            binaryHeap.Add(3);
            binaryHeap.Add(4);
            binaryHeap.Add(2);

            Console.WriteLine(binaryHeap.toString());
            binaryHeap.Pop();

            Console.WriteLine(binaryHeap.toString());
            binaryHeap.Pop();

            Console.WriteLine(binaryHeap.toString());
            binaryHeap.Pop();

            Console.WriteLine(binaryHeap.toString());
            binaryHeap.Pop();

            GraphNode node1 = new GraphNode(1);
            GraphNode node2 = new GraphNode(2);
            GraphNode node3 = new GraphNode(3);
            GraphNode node4 = new GraphNode(4);
            GraphNode node5 = new GraphNode(5);
            GraphNode node6 = new GraphNode(6);
            node1.Neighbours.Add(node2, 1);
            node1.Neighbours.Add(node3, 2);
            node2.Neighbours.Add(node4, 5);
            node3.Neighbours.Add(node5, 6);
            node4.Neighbours.Add(node6, 1);
            node5.Neighbours.Add(node6, 1);

            Dijsktra d = new Dijsktra();
            d.Solve(node1);

            GraphNode current = node6;
            while (current != null) {
                Console.WriteLine(current.Value);
                current = current.Previous;
            }

            Topological topological = new Topological();
            var result = topological.Solve(node1);

            Console.WriteLine(
                string.Join(": ",
                            result.Select(n => n.Value).ToArray()));

            Console.WriteLine("hi there");
        }
    }
}
