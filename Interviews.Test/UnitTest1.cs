using System;
using Xunit;

namespace Interviews.Test
{
    public class UnitTest1
    {
        [Fact]
        public void Test1()
        {
            int expected = 1;
            Assert.Equal(expected, Program.Solve());
        }
    }
}
