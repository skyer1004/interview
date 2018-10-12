// 1 // 681. Next Closest Time
// Given a time represented in the format "HH:MM", form the next closest time by reusing the current digits. There is no limit on how many times a digit can be reused.
class Solution {
    
    public String nextClosestTime(String time) {
        char[] result = time.toCharArray();
        char[] digits = new char[] {result[0], result[1], result[3], result[4]};
        Arrays.sort(digits);
        
        // find next digit for HH:M_
        result[4] = findNext(result[4], (char)('9' + 1), digits);  // no upperLimit for this digit, i.e. 0-9
        if(result[4] > time.charAt(4)) return String.valueOf(result);  // e.g. 23:43 -> 23:44
        
        // find next digit for HH:_M
        result[3] = findNext(result[3], '5', digits);
        if(result[3] > time.charAt(3)) return String.valueOf(result);  // e.g. 14:29 -> 14:41
        
        // find next digit for H_:MM
        result[1] = result[0] == '2' ? findNext(result[1], '3', digits) : findNext(result[1], (char)('9' + 1), digits);
        if(result[1] > time.charAt(1)) return String.valueOf(result);  // e.g. 02:37 -> 03:00 
        
        // find next digit for _H:MM
        result[0] = findNext(result[0], '2', digits);
        return String.valueOf(result);  // e.g. 19:59 -> 11:11
    }
    
    /** 
     * find the next bigger digit which is no more than upperLimit. 
     * If no such digit exists in digits[], return the minimum one i.e. digits[0]
     * @param current the current digit
     * @param upperLimit the maximum possible value for current digit
     * @param digits[] the sorted digits array
     * @return 
     */
    private char findNext(char current, char upperLimit, char[] digits) {
        //System.out.println(current);
        if(current == upperLimit) {
            return digits[0];
        }
        int pos = Arrays.binarySearch(digits, current) + 1;
        while(pos < 4 && (digits[pos] > upperLimit || digits[pos] == current)) { // traverse one by one to find next greater digit
            pos++;
        }
        return pos == 4 ? digits[0] : digits[pos];
    }
}

// 2 // 683. K Empty Slots
// There is a garden with N slots. In each slot, there is a flower. The N flowers will bloom one by one in N days. In each day, there will be exactly one flower blooming and it will be in the status of blooming since then.
// Given an array flowers consists of number from 1 to N. Each number in the array represents the place where the flower will open in that day.
// For example, flowers[i] = x means that the unique flower that blooms at day i will be at position x, where i and x will be in the range from 1 to N.
// Also given an integer k, you need to output in which day there exists two flowers in the status of blooming, and also the number of flowers between them is k and these flowers are not blooming.
// If there isnt such day, output -1.

class Solution {
    public int kEmptySlots(int[] flowers, int k) {
        int[] days =  new int[flowers.length];
        for(int i=0; i<flowers.length; i++)days[flowers[i]-1] = i + 1;
        int left = 0, right = k + 1, res = Integer.MAX_VALUE;
        for(int i = 0; right < days.length; i++){
            if(days[i] < days[left] || days[i] <= days[right]){
                if(i == right)res = Math.min(res, Math.max(days[left], days[right]));   //we get a valid subarray
                left = i; 
                right = k + 1 + i;
            }
        }
        return (res == Integer.MAX_VALUE)?-1:res;
    }
}

// 3 // 159. Longest Substring with At Most Two Distinct Characters
// Given a string s , find the length of the longest substring t  that contains at most 2 distinct characters.
class Solution {
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        char t1 = '1';
        char t2 = '1';
        int maxlen = 0;
        int count = 0;
        for(int i = 0; i < s.length(); i++){
            if(t1 == '1'){
                t1 = s.charAt(i);
                count++;
            }else if(t1 == s.charAt(i)){
                count++;
            }else if(t2 == '1'){
                t2 = s.charAt(i);
                count++;
            }else if(t2 == s.charAt(i)){
                count++;
            }else{
                if(s.charAt(i-1) == t1){
                    t2 = s.charAt(i);
                    int temp = i-1;
                    count = 1;
                    while(temp>=0 && s.charAt(temp--) == t1)count++;
                }
                else if(s.charAt(i-1) == t2){
                    t1 = s.charAt(i);
                    int temp = i-1;
                    count = 1;
                    while(temp>=0 && s.charAt(temp--) == t2)count++;
                }
            }
            if(count > maxlen) maxlen = count;
        }
        return maxlen;
    }
}

// 4 // 399. Evaluate Division
// Equations are given in the format A / B = k, where A and B are variables represented as strings, and k is a real number (floating point number). Given some queries, return the answers. If the answer does not exist, return -1.0.
class Solution {
    class Pair{
        public String word;
        public double val;
        public Pair(){}
        public Pair(String word, double val){
            this.word = word;
            this.val = val;
        }
    }
    public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
        Map<String, Pair> map = new HashMap<>();
        double[] res = new double[queries.length];
        for(int i = 0; i < equations.length; i++){
            String s1 = equations[i][0];
            String s2 = equations[i][1];
            Pair p1 = find(map, s1);
            Pair p2 = find(map, s2);
            if(!p1.word.equals(p2.word)) map.put(p1.word, new Pair(s2, values[i]/p1.val));
        }
        for(int i = 0; i < queries.length; i++){
            String q1 = queries[i][0];
            String q2 = queries[i][1];
            if(!map.containsKey(q1)||!map.containsKey(q2)){
                res[i] = -1.0;
                continue;
            }
            Pair p1 = find(map, q1);
            Pair p2 = find(map, q2);
            if(!p1.word.equals(p2.word)){
                res[i] = -1.0;
            }
            else{
                res[i] = p1.val/p2.val;
            }
        }
        return res;
    }
    public Pair find(Map<String, Pair> map, String a){
        if(!map.containsKey(a)){
            Pair pair = new Pair(a, 1.0);
            map.put(a, pair);
        }
        double value = 1;
        String b = a;
        while(map.containsKey(b) && !b.equals(map.get(b).word)){
            value*= map.get(b).val;
            b = map.get(b).word;
        }
        Pair temp = map.get(a);
        temp.val = value;
        temp.word = b;
        map.put(a, temp);
        return temp;
    }
}

// 5 // 843. Guess the Word
// This problem is an interactive problem new to the LeetCode platform.
// We are given a word list of unique words, each word is 6 letters long, and one word in this list is chosen as secret.
// You may call master.guess(word) to guess a word.  The guessed word should have type string and must be from the original list with 6 lowercase letters.
// This function returns an integer type, representing the number of exact matches (value and position) of your guess to the secret word.  Also, if your guess is not in the given wordlist, it will return -1 instead.
// For each test case, you have 10 guesses to guess the word. At the end of any number of calls, if you have made 10 or less calls to master.guess and at least one of these guesses was the secret, you pass the testcase.
// Besides the example test case below, there will be 5 additional test cases, each with 100 words in the word list.  The letters of each word in those testcases were chosen independently at random from 'a' to 'z', such that every word in the given word lists is unique.

/**
 * // This is the Master's API interface.
 * // You should not implement it, or speculate about its implementation
 * interface Master {
 *     public int guess(String word) {}
 * }
 */
class Solution {
    public void findSecretWord(String[] wordlist, Master master) {
        int[] count = new int[26];
        Set<String> wordSet = new HashSet<>();
        for(String s : wordlist){
            wordSet.add(s);
            for(char c : s.toCharArray()){
                count[c-'a'] += 1;
            }
        }
        for(int i = 0; i < 10; i++){
            String best = getBest(wordSet, count);
            int match = master.guess(best);
            if(match == 6) return;
            for(Iterator<String> it = wordSet.iterator(); it.hasNext();){
                 if(dis(it.next(), best) != match) it.remove();
            }
        }
    }
    public String getBest(Set<String> wordSet, int[] count){
        String best = "";
        int max = 0;
        for(String s : wordSet){
            int cur = 0;
            for(char c : s.toCharArray()){
                cur+= count[c - 'a'];
            }
            if(cur > max){
                max = cur;
                best = s;
            }
        }
        return best;
    }
    public int dis(String a, String b){
        int res = 0;
        for(int i = 0; i<a.length(); i++){
           if(a.charAt(i) == b.charAt(i)){
               res++;
           }
        }
        return res;
    }
}

// 6 // 857. Minimum Cost to Hire K Workers
// There are N workers.  The i-th worker has a quality[i] and a minimum wage expectation wage[i].
// Now we want to hire exactly K workers to form a paid group.  When hiring a group of K workers, we must pay them according to the following rules:
// Every worker in the paid group should be paid in the ratio of their quality compared to other workers in the paid group.
// Every worker in the paid group must be paid at least their minimum wage expectation.
// Return the least amount of money needed to form a paid group satisfying the above conditions.
class Solution {
    public double mincostToHireWorkers(int[] quality, int[] wage, int K) {
        double[][] worker = new double[quality.length][2];
        for(int i = 0; i < quality.length; i++){
            worker[i][0] = (double)wage[i]/quality[i];
            worker[i][1] = (double)quality[i];
        }
        Arrays.sort(worker, (a, b)->{return a[0] - b[0] > 0? 1 : -1;});
        double res = Integer.MAX_VALUE;
        double qsum = 0;
        PriorityQueue<Double> queue = new PriorityQueue<>();
        for(int i = 0; i < worker.length; i++){
            qsum += worker[i][1];
            queue.offer(-worker[i][1]);
            if(queue.size() > K) qsum+=queue.poll();
            if(queue.size() == K) res = Math.min(res, qsum*worker[i][0]);
        }
        return res;
    }
}

// 7 // 844. Backspace String Compare
// Given two strings S and T, return if they are equal when both are typed into empty text editors. # means a backspace character.
class Solution {
    public boolean backspaceCompare(String S, String T) {
        Stack<Character> staS = new Stack<>();
        Stack<Character> staT = new Stack<>();
        for(char c : S.toCharArray()){
            if(c != '#') staS.push(c);
            else if(!staS.isEmpty())staS.pop();
        }
        for(char c : T.toCharArray()){
            if(c != '#') staT.push(c);
            else if(!staT.isEmpty())staT.pop();
        }
        if(staS.size() != staT.size()) return false;
        while(!staS.isEmpty()){
            char s = staS.pop();
            char t = staT.pop();
            if(s != t) return false;
        }
        return true;
    }
}

// 8 // 904. Fruit Into Baskets
// In a row of trees, the i-th tree produces fruit with type tree[i].
// You start at any tree of your choice, then repeatedly perform the following steps:
// Add one piece of fruit from this tree to your baskets.  If you cannot, stop.
// Move to the next tree to the right of the current tree.  If there is no tree to the right, stop.
// Note that you do not have any choice after the initial choice of starting tree: you must perform step 1, then step 2, then back to step 1, then step 2, and so on until you stop.
// You have two baskets, and each basket can carry any quantity of fruit, but you want each basket to only carry one type of fruit each.
// What is the total amount of fruit you can collect with this procedure?

class Solution {
    public int totalFruit(int[] tree) {
        int total = -1;
        int t1=-1, t2=-1;
        int count = 0;
        for(int i = 0; i < tree.length; i++){
           if(t1 == -1){
               t1 = tree[i];
               count++;
           }else if(t1 == tree[i]){
               count++;
           }else if(t2 == -1){
               t2 = tree[i];
               count++;
           }else if(t2 == tree[i]){
               count++;
           }else{
               if(count > total) total = count;
               if(tree[i-1] == t1){
                   t2 = tree[i];
               }
               else if(tree[i-1] == t2){
                   t1 = tree[i];
               }
               int index = i-1;
               int temp = 0;
               while(tree[index--] == tree[i-1])temp++;
               count = temp;
               count++;
           }
        }
        if(count > total) total = count;
        return total;
    }
}

// 9 // 489. Robot Room Cleaner
// Given a robot cleaner in a room modeled as a grid.
// Each cell in the grid can be empty or blocked.
// The robot cleaner with 4 given APIs can move forward, turn left or turn right. Each turn it made is 90 degrees.
// When it tries to move into a blocked cell, its bumper sensor detects the obstacle and it stays on the current cell.

/**
 * // This is the robot's control interface.
 * // You should not implement it, or speculate about its implementation
 * interface Robot {
 *     // Returns true if the cell in front is open and robot moves into the cell.
 *     // Returns false if the cell in front is blocked and robot stays in the current cell.
 *     public boolean move();
 *
 *     // Robot will stay in the same cell after calling turnLeft/turnRight.
 *     // Each turn will be 90 degrees.
 *     public void turnLeft();
 *     public void turnRight();
 *
 *     // Clean the current cell.
 *     public void clean();
 * }
 */
class Solution {
    public void cleanRoom(Robot robot) {
        Set<String> path = new HashSet<>();
        solve(robot, path, 0, 0, 0);
    }
    public void solve(Robot robot, Set<String> set, int i, int j, int cur){
        String pos = i + "->" + j;
        if(set.contains(pos)) return;
        System.out.println(pos);
        robot.clean();
        set.add(pos); 
        for(int n = 0; n < 4; n++){
            if(robot.move()){
              int x = i, y = j;
                switch(cur){
                    case 0 : 
                        x = i-1; 
                        break;
                    case 90: 
                        y = j+1; 
                        break;
                    case 180:
                        x = i+1; 
                        break; 
                        
                    case 270:
                        y = j-1;
                        break;
                    default: break;
                }
                solve(robot, set, x, y, cur);  
                robot.turnLeft();
                robot.turnLeft();
                robot.move();
                robot.turnRight();
                robot.turnRight();
            }  
            robot.turnRight();
            cur+=90;
            cur = cur%360;
        }  
    }
}

// 10 // 803. Bricks Falling When Hit
// We have a grid of 1s and 0s; the 1s in a cell represent bricks.  A brick will not drop if and only if it is directly connected to the top of the grid, or at least one of its (4-way) adjacent bricks will not drop.
// We will do some erasures sequentially. Each time we want to do the erasure at the location (i, j), the brick (if it exists) on that location will disappear, and then some other bricks may drop because of that erasure.
// Return an array representing the number of bricks that will drop after each erasure in sequence.
class Solution {
    int[] father;
    int[] size;
    
    public int[] hitBricks(int[][] grid, int[][] hits) {
        int[] res = new int[hits.length];    
        int row = grid.length;
        int col = grid[0].length;
        father = new int[row*col+1];
        size = new int[row*col+1];
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                    father[i*col+j] = i*col+j;
                    size[i*col+j] = 1;
            }
        }
        for(int i = 0; i < hits.length; i++){
            int x = hits[i][0];
            int y = hits[i][1];
            if(grid[x][y] == 1) grid[x][y] = 2;
        }
        
        
        father[row*col] = row*col;
        size[row*col] = 1;
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                if(grid[i][j] == 1){
                    unionAround(grid, i, j);
                }
            }
        }
        int count = size[find(row*col)];
        for(int i = hits.length-1; i >= 0; i--){
            int x = hits[i][0];
            int y = hits[i][1];
            if(grid[x][y] == 2){
              unionAround(grid, x, y);
              grid[x][y] = 1;
              int newCount = size[find(row*col)];
              res[i] = (newCount - count > 0) ? newCount - count - 1 : 0;
              count = newCount;
            }
        }
        return res;
    }
    public void union(int pos1, int pos2){
        int f1 = find(pos1);
        int f2 = find(pos2);
        if(f1 != f2){
          father[f2] = f1;
          size[f1] += size[f2];
        }
    }
    public int find(int pos){
        int toFind = pos;
        while(father[toFind] != toFind){
            father[toFind] = father[father[toFind]];
            toFind = father[toFind];
        }
       
        return toFind;
    }
    public void unionAround(int[][] grid, int i, int j){
        int row = grid.length;
        int col = grid[0].length;
        if(isValid(grid, i, j+1)){
           
            union(i*col+j, i*col+j+1);  
        } 
        if(isValid(grid, i+1, j)){
         
          union(i*col+j, (i+1)*col+j);  
        } 
        if(isValid(grid, i, j-1)){
           
            union(i*col+j, i*col+j-1);
        }
        if(isValid(grid, i-1, j)){
          
            union(i*col+j, (i-1)*col+j);
        } 
        if(i == 0) union(row*col, i*col+j);
    }
    public boolean isValid(int[][] grid, int x, int y){
        int r = grid.length-1;
        int c = grid[0].length-1;
        if(x < 0 || x > r) return false;
        if(y < 0 || y > c) return false;
        if(grid[x][y] != 1) return false;
        return true;
    }
}

// 11 // 686. Repeated String Match
// Given two strings A and B, find the minimum number of times A has to be repeated such that B is a substring of it. If no such solution, return -1.
// For example, with A = "abcd" and B = "cdabcdab".
// Return 3, because by repeating A three times (“abcdabcdabcd”), B is a substring of it; and B is not a substring of A repeated two times ("abcdabcd").
// Note: The length of A and B will be between 1 and 10000.
class Solution {
    public int repeatedStringMatch(String A, String B) {
        if(A == null || A.length() == 0) return -1;
        if(B == null || B.length() == 0) return -1;
        if(A.equals(B)) return 1;
        if(A.indexOf(B) != -1) return 1;
        int lenA = A.length();
        int lenB = B.length();
        int len = lenA;
        int count = 2;
        while(count*lenA <= 2*lenB)count++;
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < count; i++){
            sb.append(A);
        }
        if(sb.toString().indexOf(B) == -1) return -1;
        StringBuilder res = new StringBuilder();
        for(int i = 0; i < count; i++){
            res.append(A);
            if(res.toString().indexOf(B) != -1) return i+1;
        }
        return -1;
    }
}

// 12 //299. Bulls and Cows
//You are playing the following Bulls and Cows game with your friend: You write down a number and ask your friend to guess what the number is. Each time your friend makes a guess, you provide a hint that indicates how many digits in said guess match your secret number exactly in both digit and position (called "bulls") and how many digits match the secret number but locate in the wrong position (called "cows"). Your friend will use successive guesses and hints to eventually derive the secret number.
//Write a function to return a hint according to the secret number and friend's guess, use A to indicate the bulls and B to indicate the cows. 
//Please note that both secret number and friend's guess may contain duplicate digits.
class Solution {
public:
    string getHint(string secret, string guess) {
        map<char,int> index;
        int bull=0;
        int cow=0;
        
        for(int i=0;i<secret.size();i++){
                if(secret[i]==guess[i])++bull;
                ++index[secret[i]];
                
        }
        for(int i=0;i<guess.size();i++){
            if(index[guess[i]]){
                cow++;
                --index[guess[i]];
            }
        }
        
        return to_string(bull)+"A"+to_string(cow-bull)+"B";
    }
};

// 13 //562. Longest Line of Consecutive One in Matrix （这道题没做，贴的是discussion里面点赞高的答案）
/*Given a 01 matrix M, find the longest line of consecutive one in the matrix. The line could be horizontal, vertical, diagonal or anti-diagonal.
Example:
Input:
[[0,1,1,0],
 [0,1,1,0],
 [0,0,0,1]]
Output: 3
*/
public int longestLine(int[][] M) {
    int n = M.length, max = 0;
    if (n == 0) return max;
    int m = M[0].length;
    int[][][] dp = new int[n][m][4];
    for (int i=0;i<n;i++) 
        for (int j=0;j<m;j++) {
            if (M[i][j] == 0) continue;
            for (int k=0;k<4;k++) dp[i][j][k] = 1;
            if (j > 0) dp[i][j][0] += dp[i][j-1][0]; // horizontal line
            if (j > 0 && i > 0) dp[i][j][1] += dp[i-1][j-1][1]; // anti-diagonal line
            if (i > 0) dp[i][j][2] += dp[i-1][j][2]; // vertical line
            if (j < m-1 && i > 0) dp[i][j][3] += dp[i-1][j+1][3]; // diagonal line
            max = Math.max(max, Math.max(dp[i][j][0], dp[i][j][1]));
            max = Math.max(max, Math.max(dp[i][j][2], dp[i][j][3]));
        }
    return max;
}

// 14 //135. Candy
/*There are N children standing in a line. Each child is assigned a rating value.
You are giving candies to these children subjected to the following requirements:
Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
What is the minimum candies you must give?*/
class Solution {
    public int candy(int[] ratings) {
        if(ratings == null || ratings.length == 0) return 0;
        int pre = 1, total = 1, cd = 0;
        for(int i = 1; i < ratings.length; i++){
            if(ratings[i] >= ratings[i-1]){
                if(cd > 0){
                    total+= (cd+1)*cd/2;
                    if(cd >= pre) total+= cd-pre+1;
                    cd = 0;
                    pre = 1;
                }
                pre = ratings[i] == ratings[i-1] ? 1 : pre+1;
                total+=pre;
            }
            else cd++;
        }
        if(cd > 0){
                    total+= (cd+1)*cd/2;
                    if(cd >= pre) total+= cd-pre+1;
                }
        return total;
    }
}

// 15 //308. Range Sum Query 2D - Mutable
//Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
class NumMatrix {
    
    int[][] matrix;
    int[][] dp;
    public NumMatrix(int[][] matrix) {
        this.matrix = matrix;
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0) dp = new int[0][0];
        else{
            dp = new int[matrix.length][matrix[0].length];
            init();    
        }
        
    }
    
    public void init(){
        dp[0][0] = matrix[0][0];
        for(int i = 1; i < matrix.length; i++){
            dp[i][0] = dp[i-1][0] + matrix[i][0];
        }
        for(int j = 1; j < matrix[0].length; j++){
            dp[0][j] = dp[0][j-1] + matrix[0][j];
        }
        for(int i = 1; i < matrix.length; i++){
            for(int j = 1; j < matrix[0].length; j++){
                dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + matrix[i][j];
            }
        }
    }
    public void update(int row, int col, int val) {
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0) return;
        int num = matrix[row][col];
        matrix[row][col] = val;
        int delta = val - num;
        for(int i = row; i < matrix.length; i++){
            for(int j = col; j < matrix[0].length;j++){
                dp[i][j]+=delta;
            }
        }

    }
    
    public int sumRegion(int row1, int col1, int row2, int col2) {
         if(matrix == null || matrix.length == 0 || matrix[0].length == 0) return 0;
         int s2 = (row2>=0 && row2 < matrix.length && col2 >=0 && col2 < matrix[0].length)?dp[row2][col2]: 0;
         int t1 = (row1 > 0 && row1 < matrix.length && col2 >=0 && col2 < matrix[0].length)?dp[row1-1][col2]: 0;
         int t2 = (row2 >=0 && row2 < matrix.length && col1 > 0 && col1 < matrix[0].length)?dp[row2][col1-1]: 0;
         int s1 = (row1 > 0 && row1 < matrix.length && col1 > 0 && col1 < matrix[0].length)?dp[row1-1][col1-1]: 0;
         return s2-t1-t2+s1;
    }
}

// 16 
/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix obj = new NumMatrix(matrix);
 * obj.update(row,col,val);
 * int param_2 = obj.sumRegion(row1,col1,row2,col2);
 */



// 1. Two Sum
// Given an array of integers, return indices of the two numbers such that they add up to a specific target.
// You may assume that each input would have exactly one solution, and you may not use the same element twice.
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        if(nums == null || nums.length == 0) return res;
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            if(map.containsKey(target - nums[i])){
                res[0] =map.get(target - nums[i]);
                res[1] = i;
                return res;
            }
            else{
                map.put(nums[i], i);
            }
        }
        return res;
    }
}

// 17 // 403. Frog Jump
// A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.
// Given a list of stones' positions (in units) in sorted ascending order, determine if the frog is able to cross the river by landing on the last stone. Initially, the frog is on the first stone and assume the first jump must be 1 unit.
// If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units. Note that the frog can only jump in the forward direction.
// Note:
// The number of stones is ≥ 2 and is < 1,100.
// Each stone's position will be a non-negative integer < 231.
// The first stone's position is always 0.
class Solution {
    public boolean canCross(int[] stones) {
        Map<Integer, Set<Integer>> dp = new HashMap<>();
        for(int stone : stones){
            dp.put(stone, new HashSet<Integer>());
        }
        dp.get(0).add(1);
       for(int stone : stones){
           for(int step : dp.get(stone)){
               int dis = step + stone;
               if(dis == stones[stones.length-1]) return true;
               if(dp.containsKey(dis)){
                   Set<Integer> set = dp.get(dis);
                   set.add(step);
                   if(step > 1) set.add(step-1);
                   set.add(step+1);
               }
           }
       }
       return false;
    }
}


// 18 // 465. Optimal Account Balancing
// A group of friends went on holiday and sometimes lent each other money. For example, Alice paid for Bill's lunch for $10. Then later Chris gave Alice $5 for a taxi ride. We can model each transaction as a tuple (x, y, z) which means person x gave person y $z. Assuming Alice, Bill, and Chris are person 0, 1, and 2 respectively (0, 1, 2 are the person's ID), the transactions can be represented as [[0, 1, 10], [2, 0, 5]].
// Given a list of transactions between a group of people, return the minimum number of transactions required to settle the debt.
// Note:
// A transaction will be given as a tuple (x, y, z). Note that x ≠ y and z > 0.
// Person's IDs may not be linear, e.g. we could have the persons 0, 1, 2 or we could also have the persons 0, 2, 6.
class Solution {
    public int minTransfers(int[][] trans) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < trans.length; i++){
            int per1 = trans[i][0];
            int per2 = trans[i][1];
            int val = trans[i][2];
            if(map.containsKey(per1)){
                map.put(per1, map.get(per1) - val);
            }
            else{
                map.put(per1, -val);
            }
            if(map.containsKey(per2)){
                map.put(per2, map.get(per2) + val);
            }else{
                map.put(per2, val);
            }
        }
        List<Integer> poslist = new ArrayList<>();
        List<Integer> neglist = new ArrayList<>();
        for(int x : map.keySet()){
            if(map.get(x) < 0)
                neglist.add(-map.get(x));
            else
                poslist.add(map.get(x));
        }
        int min = Integer.MAX_VALUE;
       
        Stack<Integer> psta = new Stack<>();
        Stack<Integer> nsta = new Stack<>();
        for(int i = 0; i < 1000; i++){
            for(int x : poslist)psta.push(x);
            for(int x : neglist)nsta.push(x);
            int cur = 0;
            while(!nsta.empty()){
                int pos = psta.pop();
                int neg = nsta.pop();
                cur++;
                if(pos == neg) continue;
                if(pos > neg){
                    psta.push(pos - neg);
                }
                else{
                    nsta.push(neg-pos);
                }
            }
            if(cur < min) min = cur;
            Collections.shuffle(poslist);
            Collections.shuffle(neglist);
        }
        return min;
    }
}

// 19 // 642. Design Search Autocomplete System
// Design a search autocomplete system for a search engine. Users may input a sentence (at least one word and end with a special character '#'). For each character they type except '#', you need to return the top 3 historical hot sentences that have prefix the same as the part of sentence already typed. Here are the specific rules:
// The hot degree for a sentence is defined as the number of times a user typed the exactly same sentence before.
// The returned top 3 hot sentences should be sorted by hot degree (The first is the hottest one). If several sentences have the same degree of hot, you need to use ASCII-code order (smaller one appears first).
// If less than 3 hot sentences exist, then just return as many as you can.
// When the input is a special character, it means the sentence ends, and in this case, you need to return an empty list.
// Your job is to implement the following functions:
// The constructor function:
// AutocompleteSystem(String[] sentences, int[] times): This is the constructor. The input is historical data. Sentences is a string array consists of previously typed sentences. Times is the corresponding times a sentence has been typed. Your system should record these historical data.
// Now, the user wants to input a new sentence. The following function will provide the next character the user types:
// List<String> input(char c): The input c is the next character typed by the user. The character will only be lower-case letters ('a' to 'z'), blank space (' ') or a special character ('#'). Also, the previously typed sentence should be recorded in your system. The output will be the top 3 historical hot sentences that have prefix the same as the part of sentence already typed.

public class AutocompleteSystem {
    class TrieNode {
        Map<Character, TrieNode> children;
        Map<String, Integer> counts;
        boolean isWord;
        public TrieNode() {
            children = new HashMap<Character, TrieNode>();
            counts = new HashMap<String, Integer>();
            isWord = false;
        }
    }
    
    class Pair {
        String s;
        int c;
        public Pair(String s, int c) {
            this.s = s; this.c = c;
        }
    }
    
    TrieNode root;
    String prefix;
    
    
    public AutocompleteSystem(String[] sentences, int[] times) {
        root = new TrieNode();
        prefix = "";
        
        for (int i = 0; i < sentences.length; i++) {
            add(sentences[i], times[i]);
        }
    }
    
    private void add(String s, int count) {
        TrieNode curr = root;
        for (char c : s.toCharArray()) {
            TrieNode next = curr.children.get(c);
            if (next == null) {
                next = new TrieNode();
                curr.children.put(c, next);
            }
            curr = next;
            curr.counts.put(s, curr.counts.getOrDefault(s, 0) + count);
        }
        curr.isWord = true;
    }
    
    public List<String> input(char c) {
        if (c == '#') {
            add(prefix, 1);
            prefix = "";
            return new ArrayList<String>();
        }
        
        prefix = prefix + c;
        TrieNode curr = root;
        for (char cc : prefix.toCharArray()) {
            TrieNode next = curr.children.get(cc);
            if (next == null) {
                return new ArrayList<String>();
            }
            curr = next;
        }
        
        PriorityQueue<Pair> pq = new PriorityQueue<>((a, b) -> (a.c == b.c ? a.s.compareTo(b.s) : b.c - a.c));
        for (String s : curr.counts.keySet()) {
            pq.add(new Pair(s, curr.counts.get(s)));
        }

        List<String> res = new ArrayList<String>();
        for (int i = 0; i < 3 && !pq.isEmpty(); i++) {
            res.add(pq.poll().s);
        }
        return res;
    }
}

/**
 * Your AutocompleteSystem object will be instantiated and called as such:
 * AutocompleteSystem obj = new AutocompleteSystem(sentences, times);
 * List<String> param_1 = obj.input(c);
 */

// 20 // 428. Serialize and Deserialize N-ary Tree
// Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
// Design an algorithm to serialize and deserialize an N-ary tree. An N-ary tree is a rooted tree in which each node has no more than N children. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that an N-ary tree can be serialized to a string and this string can be deserialized to the original tree structure.
class Codec {

    // Encodes a tree to a single string.
    public String serialize(Node root) {
        List<String> list=new LinkedList<>();
        serializeHelper(root,list);
        return String.join(",",list);
    }
    
    private void serializeHelper(Node root, List<String> list){
        if(root==null){
            return;
        }else{
            list.add(String.valueOf(root.val));
            list.add(String.valueOf(root.children.size()));
            for(Node child:root.children){
                serializeHelper(child,list);
            }
        }
    }

    // Decodes your encoded data to tree.
    public Node deserialize(String data) {
        if(data.isEmpty())
            return null;
        
        String[] ss=data.split(",");
        Queue<String> q=new LinkedList<>(Arrays.asList(ss));
        return deserializeHelper(q);
    }
    
    private Node deserializeHelper(Queue<String> q){
        Node root=new Node();
        root.val=Integer.parseInt(q.poll());
        int size=Integer.parseInt(q.poll());
        root.children=new ArrayList<Node>(size);
        for(int i=0;i<size;i++){
            root.children.add(deserializeHelper(q));
        }
        return root;
    }
}

// 21 // 846. Hand of Straights （没做过）
// Alice has a hand of cards, given as an array of integers.
// Now she wants to rearrange the cards into groups so that each group is size W, and consists of W consecutive cards.
// Return true if and only if she can.
class Solution {
    public boolean isNStraightHand(int[] hand, int W) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for(int i : hand){
            minHeap.add(i);
        }
        while(minHeap.size() != 0) {
            int start = minHeap.poll();
            for(int j = 1; j < W; j++){
                if(minHeap.remove(start + j)) {
                    continue;
                }
                else {
                    return false;
                }
            }
        }
        return true;
    }
}

// 22 // 818. Race Car
// Your car starts at position 0 and speed +1 on an infinite number line.  (Your car can go into negative positions.)
// Your car drives automatically according to a sequence of instructions A (accelerate) and R (reverse).
// When you get an instruction "A", your car does the following: position += speed, speed *= 2.
// When you get an instruction "R", your car does the following: if your speed is positive then speed = -1 , otherwise speed = 1.  (Your position stays the same.)
// For example, after commands "AAR", your car goes to positions 0->1->3->3, and your speed goes to 1->2->4->-1.
class Solution {
    public int racecar(int target) {
        int[] dp = new int[10001];
        for(int i=0; i <= 13; i++){
            dp[(1<<i)-1] = i;
        }
        solve(target, dp);
        return dp[target];
    }
    public int solve(int target, int[] dp){
        if(dp[target] != 0) return dp[target];
        int n = (int)(Math.log(target)/Math.log(2))+1;
        //scenario 1
        dp[target] = n + 1 + solve((1<<n) - 1 - target, dp);
        //scenario 2
        for(int m = 0; m < n-1; m++){
            dp[target] = Math.min(dp[target], n + m + 1 + solve(target - (1<<n -1) + (1<<m), dp));
        }
        return dp[target];
    }
}

// 23 // 205. Isomorphic Strings
// Given two strings s and t, determine if they are isomorphic.
// Two strings are isomorphic if the characters in s can be replaced to get t.
// All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character but a character may map to itself.
class Solution {
    public boolean isIsomorphic(String s, String t) {
        if(s.length() != t.length()) return false;
        int count = 1;
        Map<Character, Integer> map1 = new HashMap<>();
        Map<Character, Integer> map2 = new HashMap<>();
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        for(char c : s.toCharArray()){
            if(map1.containsKey(c)){
                sb1.append(map1.get(c));
            }
            else{
                map1.put(c, count);
                sb1.append(count);
                count++;
            }
        }
        count = 1;
        for(char c : t.toCharArray()){
            if(map2.containsKey(c)){
                sb2.append(map2.get(c));
            }
            else{
                map2.put(c, count);
                sb2.append(count);
                count++;
            }
        }
        return sb1.toString().equals(sb2.toString());
    }
}

// 24 // 766. Toeplitz Matrix
// A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same element.
// Now given an M x N matrix, return True if and only if the matrix is Toeplitz.
class Solution {
    public boolean isToeplitzMatrix(int[][] matrix) {
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[0].length; j++){
                if(i-1 >= 0 && j-1>=0 && matrix[i][j] != matrix[i-1][j-1]) return false;
                if(i+1 < matrix.length && j+1 < matrix[0].length && matrix[i][j] != matrix[i+1][j+1]) return false;
            }
        }
        return true;
    }
}

// 25 // 833. Find And Replace in String
// To some string S, we will perform some replacement operations that replace groups of letters with new ones (not necessarily the same size).
// Each replacement operation has 3 parameters: a starting index i, a source word x and a target word y.  The rule is that if x starts at position i in the original string S, then we will replace that occurrence of x with y.  If not, we do nothing.
// For example, if we have S = "abcd" and we have some replacement operation i = 2, x = "cd", y = "ffff", then because "cd" starts at position 2 in the original string S, we will replace it with "ffff".
// Using another example on S = "abcd", if we have both the replacement operation i = 0, x = "ab", y = "eee", as well as another replacement operation i = 2, x = "ec", y = "ffff", this second operation does nothing because in the original string S[2] = 'c', which doesn't match x[0] = 'e'.
// All these operations occur simultaneously.  It's guaranteed that there won't be any overlap in replacement: for example, S = "abc", indexes = [0, 1], sources = ["ab","bc"] is not a valid test case.
class Solution {
    public String findReplaceString(String S, int[] index, String[] source, String[] targets) {
       Map<Integer, Integer> map = new HashMap<>();
       for(int i = 0; i < index.length; i++){
           if(S.startsWith(source[i], index[i])){
               map.put(index[i], i);
           }
       }
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < S.length();){
            if(map.containsKey(i)){
                sb.append(targets[map.get(i)]);
                i+=source[map.get(i)].length();
            }
            else{
                sb.append(S.charAt(i));
                i++;
            }
        }
        return sb.toString();
    }
}

// 26 // 815. Bus Routes
// We have a list of bus routes. Each routes[i] is a bus route that the i-th bus repeats forever. For example if routes[0] = [1, 5, 7], this means that the first bus (0-th indexed) travels in the sequence 1->5->7->1->5->7->1->... forever.
// We start at bus stop S (initially not on a bus), and we want to go to bus stop T. Travelling by buses only, what is the least number of buses we must take to reach our destination? Return -1 if it is not possible.
class Solution {
    public int numBusesToDestination(int[][] routes, int S, int T) {
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for(int i=0; i < routes.length; i++){
           for(int j = 0; j < routes[i].length; j++){
               if(map.containsKey(routes[i][j])){
                   Set<Integer> list = map.get(routes[i][j]);
                   list.add(i);
                   map.put(routes[i][j], list);
               }
               else{
                   Set<Integer> list = new HashSet<>();
                   list.add(i);
                   map.put(routes[i][j], list);
               }
           }
        }
        Set<Integer> visit = new HashSet<>();
        Set<Integer> station = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        if(!map.containsKey(S)) return -1;
        if(!map.containsKey(T)) return -1;
        queue.offer(S);
        int count = 0;
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0; i < size; i++){
                int cur = queue.poll();
                if(!map.containsKey(cur))continue;
                if(cur == T) return count;
                Set<Integer> buses = map.get(cur);
                for(int bus : buses){
                    if(visit.contains(bus)) continue;
                    visit.add(bus);
                    for(int route : routes[bus]){
                        if(station.contains(route)) continue;
                        station.add(route);
                        queue.offer(route);
                    }
                }
            }
            count++;
        }
        return -1;
    }
}

// 27 // 849. Maximize Distance to Closest Person
// In a row of seats, 1 represents a person sitting in that seat, and 0 represents that the seat is empty. 
// There is at least one empty seat, and at least one person sitting.
// Alex wants to sit in the seat such that the distance between him and the closest person to him is maximized. 
// Return that maximum distance to closest person.
class Solution {
    public int maxDistToClosest(int[] seats) {
        int count = 0;
        int maxlen = 0;
        if(seats[0] == 0){
            int temp = 0;
            while(seats[temp++] == 0) maxlen++;
            maxlen*=2;
        }
        for(int x : seats){
            if(x == 1){
                if(count > maxlen) maxlen = count;
                count = 0;
            }
            else count++;
        }
        if(count*2 > maxlen) maxlen = count*2;
        if(maxlen%2 == 1) return maxlen/2+1;
        else return maxlen/2;
    }
}

// 28 // 774. Minimize Max Distance to Gas Station
// On a horizontal number line, we have gas stations at positions stations[0], stations[1], ..., stations[N-1], where N = stations.length.
// Now, we add K more gas stations so that D, the maximum distance between adjacent gas stations, is minimized.
// Return the smallest possible value of D.
class Solution {
    public double minmaxGasDist(int[] stations, int K) {
        Arrays.sort(stations);
        PriorityQueue<Interval> queue = new PriorityQueue<>(new Comparator<Interval>(){
            public int compare(Interval a, Interval b){
                double diff = b.getDis() - a.getDis();
                if(diff > 0) return 1;
                if(diff < 0) return -1;
                return 0;
            }
        });
        int longest = stations[stations.length-1] - stations[0];
        int remain = K;
        for(int i = 0; i < stations.length-1; i++){
            int insertNum = (int)(K*(((double)(stations[i+1] - stations[i]))/longest));
            queue.offer(new Interval(stations[i], stations[i+1], insertNum));
            remain -= insertNum;
        }
        while(remain > 0){
            Interval temp = queue.poll();
            temp.insertNum++;
            remain--;
            queue.offer(temp);
        }
        Interval last = queue.poll();
        return last.getDis();
    }
    class Interval{
        int left;
        int right;
        int insertNum;
        Interval(int a, int b, int c){
            left = a;
            right = b;
            insertNum = c;
        }
        public double getDis(){
            return (right - left)/((double)(insertNum+1));
        }
    }
}

// 29 // 418. Sentence Screen Fitting
// Given a rows x cols screen and a sentence represented by a list of non-empty words, find how many times the given sentence can be fitted on the screen.
// Note:
// A word cannot be split into two lines.
// The order of words in the sentence must remain unchanged.
// Two consecutive words in a line must be separated by a single space.
// Total words in the sentence won't exceed 100.
// Length of each word is greater than 0 and won't exceed 10.
// 1 ≤ rows, cols ≤ 20,000.
class Solution {
    public int wordsTyping(String[] sentence, int rows, int cols) {
        int n = sentence.length;
        int[] dp = new int[n];
        
        for(int i = 0; i < n; i++) {
            int length = 0, words = 0, index = i;
            while(length + sentence[index % n].length() <= cols) {
                length += sentence[index % n].length();
                length += 1; // space
                index++;
                words++;
            }
            dp[i] = words;
        }
        
        int words = 0;
        for(int i = 0, index = 0; i < rows; i++) {
            words += dp[index];
            index = (dp[index] + index) % n;
        }
        
        return words / n;
    }
}

// 30 // 684. Redundant Connection
// In this problem, a tree is an undirected graph that is connected and has no cycles.
// The given input is a graph that started as a tree with N nodes (with distinct values 1, 2, ..., N), with one additional edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.
// The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [u, v] with u < v, that represents an undirected edge connecting nodes u and v.
// Return an edge that can be removed so that the resulting graph is a tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array. The answer edge [u, v] should be in the same format, with u < v.
class Solution {
    public int[] findRedundantConnection(int[][] edges) {
        if(edges == null || edges.length == 0) return new int[2];
        int[] father = new int[edges.length*2+1];
        for(int i = 0; i < father.length; i++){
            father[i] = i;
        }
        for(int[] edge : edges){
            int fa1 = find(edge[0], father);
            int fa2 = find(edge[1], father);
            if(fa1 == fa2) return edge;
            father[fa1] = fa2;
        }
        return new int[2];
    }
    public int find(int a, int[] father){
        if(father[a] == a) return a;
        return find(father[a], father);
    }
}

// 31 // 457. Circular Array Loop (没做过)
// You are given an array of positive and negative integers. If a number n at an index is positive, then move forward n steps. Conversely, if it's negative (-n), move backward n steps. Assume the first element of the array is forward next to the last element, and the last element is backward next to the first element. Determine if there is a loop in this array. A loop starts and ends at a particular index with more than 1 element along the loop. The loop must be "forward" or "backward'.
// Example 1: Given the array [2, -1, 1, 2, 2], there is a loop, from index 0 -> 2 -> 3 -> 0.
// Example 2: Given the array [-1, 2], there is no loop.
// Note: The given array is guaranteed to contain no element "0".
// Can you do it in O(n) time complexity and O(1) space complexity?
public class Solution {
    public boolean circularArrayLoop(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 0) {
                continue;
            }
            // slow/fast pointer
            int j = i, k = getIndex(i, nums);
            while (nums[k] * nums[i] > 0 && nums[getIndex(k, nums)] * nums[i] > 0) {
                if (j == k) {
                    // check for loop with only one element
                    if (j == getIndex(j, nums)) {
                        break;
                    }
                    return true;
                }
                j = getIndex(j, nums);
                k = getIndex(getIndex(k, nums), nums);
            }
            // loop not found, set all element along the way to 0
            j = i;
            int val = nums[i];
            while (nums[j] * val > 0) {
                int next = getIndex(j, nums);
                nums[j] = 0;
                j = next;
            }
        }
        return false;
    }
    
    public int getIndex(int i, int[] nums) {
        int n = nums.length;
        return i + nums[i] >= 0? (i + nums[i]) % n: n + ((i + nums[i]) % n);
    }
}

// 32 // 850. Rectangle Area II
// We are given a list of (axis-aligned) rectangles.  Each rectangle[i] = [x1, y1, x2, y2] , where (x1, y1) are the coordinates of the bottom-left corner, and (x2, y2) are the coordinates of the top-right corner of the ith rectangle.
// Find the total area covered by all rectangles in the plane.  Since the answer may be too large, return it modulo 10^9 + 7.

class Solution {
    
    class Point{
            int x;
            int y;
            int val;
            public Point(int a, int b, int c){
                x = a;
                y = b;
                val = c;
            }
        }
    public int rectangleArea(int[][] rectangles) {
        int M = 1000000007;
        List<Point> list = new ArrayList<>();
        for(int[] r : rectangles){
            list.add(new Point(r[0], r[1], 1));
            list.add(new Point( r[0], r[3], -1));
            list.add(new Point(r[2], r[3], 1));
            list.add(new Point(r[2], r[1], -1));
        }
        Collections.sort(list, (a, b)->{
            if(a.x == b.x)
                return a.y - b.y;
            return a.x - b.x;
        });
        TreeMap<Integer, Integer> map = new TreeMap<>();
        int res =0;
        int preX = -1;
        int preY = -1;
        for(int i= 0; i < list.size(); i++){
            Point p = list.get(i);
            map.put(p.y, map.getOrDefault(p.y, 0) + p.val);
            if(i == list.size()-1 || list.get(i).x < list.get(i+1).x){
                if(preX > -1){
                  res += ((long)preY*(p.x - preX))%M;
                  res%=M;  
                }
                preY = calY(map);
                preX = p.x;
            }
            
        }
        return res;
    }
    public int calY(TreeMap<Integer, Integer> map){
        int res = 0, pre = -1, count = 0;
        for(Map.Entry<Integer, Integer> e : map.entrySet()){
            if(pre >= 0 && count > 0){
                res+= e.getKey() - pre;    
            }
            count+=e.getValue();
            pre = e.getKey();
        }
        return res;
    }
}

// 33 // 753. Cracking the Safe
// There is a box protected by a password. The password is n digits, where each letter can be one of the first k digits 0, 1, ..., k-1.
// You can keep inputting the password, the password will automatically be matched against the last n digits entered.
// For example, assuming the password is "345", I can open it when I type "012345", but I enter a total of 6 digits.
// Please return any string of minimum length that is guaranteed to open the box after the entire string is inputted.
class Solution {
    public String crackSafe(int n, int k) {
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < n; i++)sb.append(0);
        Set<String> visit = new HashSet<>();
        visit.add(sb.toString());
        int total = (int)Math.pow(k, n);
        dfs(visit, total, sb, k, n);
        return sb.toString();
    }
    public boolean dfs(Set<String> visit, int total, StringBuilder sb, int k, int n){
        if(visit.size() == total) return true;
        String prev = sb.substring(sb.length()-n+1, sb.length());
        for(int i = 0; i < k; i++){
            String cur = prev + i;
            if(!visit.contains(cur)){
                visit.add(cur);
                sb.append(i);
                if(dfs(visit, total, sb, k,n))  return true;
                visit.remove(cur);
                sb.delete(sb.length()-1, sb.length());
            }
        }
        return false;
    }
}

// 34 // 568. Maximum Vacation Days
// LeetCode wants to give one of its best employees the option to travel among N cities to collect algorithm problems. But all work and no play makes Jack a dull boy, you could take vacations in some particular cities and weeks. Your job is to schedule the traveling to maximize the number of vacation days you could take, but there are certain rules and restrictions you need to follow.
// Rules and restrictions:
// You can only travel among N cities, represented by indexes from 0 to N-1. Initially, you are in the city indexed 0 on Monday.
// The cities are connected by flights. The flights are represented as a N*N matrix (not necessary symmetrical), called flights representing the airline status from the city i to the city j. If there is no flight from the city i to the city j, flights[i][j] = 0; Otherwise, flights[i][j] = 1. Also, flights[i][i] = 0 for all i.
// You totally have K weeks (each week has 7 days) to travel. You can only take flights at most once per day and can only take flights on each week's Monday morning. Since flight time is so short, we don't consider the impact of flight time.
// For each city, you can only have restricted vacation days in different weeks, given an N*K matrix called days representing this relationship. For the value of days[i][j], it represents the maximum days you could take vacation in the city i in the week j.
// You're given the flights matrix and days matrix, and you need to output the maximum vacation days you could take during K weeks.
class Solution {
    public int maxVacationDays(int[][] flights, int[][] days) {
        int city = days.length;
        int week = days[0].length;
        int[][] dp = new int[city][week];
        dp[0][0] = days[0][0];
        for(int i = 1; i < city; i++){
            if(flights[0][i] == 1)
                dp[i][0] = days[i][0];
            else 
                dp[i][0] = Integer.MIN_VALUE;
        }
        for(int i = 1; i < week; i++){
            for(int j = 0; j < city; j++){
                dp[j][i] = Integer.MIN_VALUE;
                if(dp[j][i-1] != Integer.MIN_VALUE)
                    dp[j][i] = dp[j][i-1] + days[j][i];
                for(int k = 0; k < city; k++){
                    if(flights[k][j] == 1){
                        dp[j][i] = Math.max(dp[j][i], dp[k][i-1] + days[j][i]);
                    }
                }
            }
        }
        int res = 0;
        for(int i = 0; i < city; i++){
            if(res < dp[i][week-1]){
                res = dp[i][week-1];
            }
        }
        return res;
    }
}

// 35 // 679. 24 Game
// You have 4 cards each containing a number from 1 to 9. You need to judge whether they could operated through *, /, +, -, (, ) to get the value of 24.
class Solution {
    public boolean judgePoint24(int[] nums) {
        List<Double> list = new ArrayList<>();
        for(int i = 0; i < nums.length; i++){
            list.add((double)nums[i]);
        }
        return dfs(list);
    }
    public boolean dfs(List<Double> list){
        if(list.size()==1){
            if(Math.abs(list.get(0) - 24) < 0.001)
                return true;
            else return false;
        }
        for(int i = 0; i < list.size(); i++){
            for(int j = i+1; j < list.size(); j++){
                for(double c : generateRes(list.get(i), list.get(j))){
                    List<Double> next = new ArrayList<>();
                    next.add(c);
                    for(int k = 0; k < list.size(); k++){
                        if(k == i || k == j)continue;
                        next.add(list.get(k));
                    }
                    if(dfs(next)) return true;
                }
            }
        }
        return false;
    }
    public List<Double> generateRes(double a, double b){
        List<Double> res = new ArrayList<>();
        res.add(a+b);
        res.add(a-b);
        res.add(b-a);
        res.add(a*b);
        res.add(a/b);
        res.add(b/a);
        return res;
    }
}

// 36 // 776. Split BST
// Given a Binary Search Tree (BST) with root node root, and a target value V, split the tree into two subtrees where one subtree has nodes that are all smaller or equal to the target value, while the other subtree has all nodes that are greater than the target value.  It's not necessarily the case that the tree contains a node with value V.
// Additionally, most of the structure of the original tree should remain.  Formally, for any child C with parent P in the original tree, if they are both in the same subtree after the split, then node C should still have the parent P.
// You should output the root TreeNode of both subtrees after splitting, in any order.
class Solution {
    public TreeNode[] splitBST(TreeNode root, int V) {
        if(root==null) return new TreeNode[]{null, null};
        
        TreeNode[] splitted;
        if(root.val<= V) {
            splitted = splitBST(root.right, V);
            root.right = splitted[0];
            splitted[0] = root;
        } else {
            splitted = splitBST(root.left, V);
            root.left = splitted[1];
            splitted[1] = root;
        }
        
        return splitted;
    }
    
}

// 37 // 447. Number of Boomerangs
// Given n points in the plane that are all pairwise distinct, a "boomerang" is a tuple of points (i, j, k) such that the distance between i and j equals the distance between i and k (the order of the tuple matters).
// Find the number of boomerangs. You may assume that n will be at most 500 and coordinates of points are all in the range [-10000, 10000] (inclusive).
class Solution {
    public int numberOfBoomerangs(int[][] points) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        for(int i = 0; i < points.length; i++){
            for(int j = 0; j < points.length; j++)
            {
                if(i == j) continue;
                int dis = getDis(points[i], points[j]);
                map.put(dis, map.getOrDefault(dis, 0)+1);
            }
            for(int val : map.values()){
                res+= val*(val-1);
            }
            map.clear();
        }
        return res;
    }
    public int getDis(int[] a, int[] b){
        int dx = a[0] - b[0];
        int dy = a[1] - b[1];
        return dx*dx + dy*dy;
    }
}

// 38 // 307. Range Sum Query - Mutable
// Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
// The update(i, val) function modifies nums by updating the element at index i to val.
public class NumArray {
	/**
	 * Binary Indexed Trees (BIT or Fenwick tree):
	 * https://www.topcoder.com/community/data-science/data-science-
	 * tutorials/binary-indexed-trees/
	 * 
	 * Example: given an array a[0]...a[7], we use a array BIT[9] to
	 * represent a tree, where index [2] is the parent of [1] and [3], [6]
	 * is the parent of [5] and [7], [4] is the parent of [2] and [6], and
	 * [8] is the parent of [4]. I.e.,
	 * 
	 * BIT[] as a binary tree:
	 *            ______________*
	 *            ______*
	 *            __*     __*
	 *            *   *   *   *
	 * indices: 0 1 2 3 4 5 6 7 8
	 * 
	 * BIT[i] = ([i] is a left child) ? the partial sum from its left most
	 * descendant to itself : the partial sum from its parent (exclusive) to
	 * itself. (check the range of "__").
	 * 
	 * Eg. BIT[1]=a[0], BIT[2]=a[1]+BIT[1]=a[1]+a[0], BIT[3]=a[2],
	 * BIT[4]=a[3]+BIT[3]+BIT[2]=a[3]+a[2]+a[1]+a[0],
	 * BIT[6]=a[5]+BIT[5]=a[5]+a[4],
	 * BIT[8]=a[7]+BIT[7]+BIT[6]+BIT[4]=a[7]+a[6]+...+a[0], ...
	 * 
	 * Thus, to update a[1]=BIT[2], we shall update BIT[2], BIT[4], BIT[8],
	 * i.e., for current [i], the next update [j] is j=i+(i&-i) //double the
	 * last 1-bit from [i].
	 * 
	 * Similarly, to get the partial sum up to a[6]=BIT[7], we shall get the
	 * sum of BIT[7], BIT[6], BIT[4], i.e., for current [i], the next
	 * summand [j] is j=i-(i&-i) // delete the last 1-bit from [i].
	 * 
	 * To obtain the original value of a[7] (corresponding to index [8] of
	 * BIT), we have to subtract BIT[7], BIT[6], BIT[4] from BIT[8], i.e.,
	 * starting from [idx-1], for current [i], the next subtrahend [j] is
	 * j=i-(i&-i), up to j==idx-(idx&-idx) exclusive. (However, a quicker
	 * way but using extra space is to store the original array.)
	 */

	int[] nums;
	int[] BIT;
	int n;

	public NumArray(int[] nums) {
		this.nums = nums;

		n = nums.length;
		BIT = new int[n + 1];
		for (int i = 0; i < n; i++)
			init(i, nums[i]);
	}

	public void init(int i, int val) {
		i++;
		while (i <= n) {
			BIT[i] += val;
			i += (i & -i);
		}
	}

	void update(int i, int val) {
		int diff = val - nums[i];
		nums[i] = val;
		init(i, diff);
	}

	public int getSum(int i) {
		int sum = 0;
		i++;
		while (i > 0) {
			sum += BIT[i];
			i -= (i & -i);
		}
		return sum;
	}

	public int sumRange(int i, int j) {
		return getSum(j) - getSum(i - 1);
	}
}

// Your NumArray object will be instantiated and called as such:
// NumArray numArray = new NumArray(nums);
// numArray.sumRange(0, 1);
// numArray.update(1, 10);
// numArray.sumRange(1, 2);

// 39 // 549. Binary Tree Longest Consecutive Sequence II
// Given a binary tree, you need to find the length of Longest Consecutive Path in Binary Tree.
// Especially, this path can be either increasing or decreasing. For example, [1,2,3,4] and [4,3,2,1] are both considered valid, but the path [1,2,4,3] is not valid. On the other hand, the path can be in the child-Parent-child order, where not necessarily be parent-child order.
public class Solution {
    int maxval = 0;
    public int longestConsecutive(TreeNode root) {
        longestPath(root);
        return maxval;
    }
    public int[] longestPath(TreeNode root) {
        if (root == null)
            return new int[] {0,0};
        int inr = 1, dcr = 1;
        if (root.left != null) {
            int[] l = longestPath(root.left);
            if (root.val == root.left.val + 1)
                dcr = l[1] + 1;
            else if (root.val == root.left.val - 1)
                inr = l[0] + 1;
        }
        if (root.right != null) {
            int[] r = longestPath(root.right);
            if (root.val == root.right.val + 1)
                dcr = Math.max(dcr, r[1] + 1);
            else if (root.val == root.right.val - 1)
                inr = Math.max(inr, r[0] + 1);
        }
        maxval = Math.max(maxval, dcr + inr - 1);
        return new int[] {inr, dcr};
    }
}

// 40 // 486. Predict the Winner
// Given an array of scores that are non-negative integers. Player 1 picks one of the numbers from either end of the array followed by the player 2 and then player 1 and so on. Each time a player picks a number, that number will not be available for the next player. This continues until all the scores have been chosen. The player with the maximum score wins.
// Given an array of scores, predict whether player 1 is the winner. You can assume each player plays to maximize his score.
public boolean PredictTheWinner(int[] nums) {
    int n = nums.length;
    int[][] dp = new int[n][n];
    for (int i = 0; i < n; i++) { dp[i][i] = nums[i]; }
    for (int len = 1; len < n; len++) {
        for (int i = 0; i < n - len; i++) {
            int j = i + len;
            dp[i][j] = Math.max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
        }
    }
    return dp[0][n - 1] >= 0;
}
// Here is the code for O(N) space complexity:
public boolean PredictTheWinner(int[] nums) {
    if (nums == null) { return true; }
    int n = nums.length;
    if ((n & 1) == 0) { return true; } // Improved with hot13399's comment.
    int[] dp = new int[n];
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                dp[i] = nums[i];
            } else {
                dp[j] = Math.max(nums[i] - dp[j], nums[j] - dp[j - 1]);
            }
        }
    }
    return dp[n - 1] >= 0;
}

// 41 // 346. Moving Average from Data Stream
// Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.
// Example:
// MovingAverage m = new MovingAverage(3);
// m.next(1) = 1
// m.next(10) = (1 + 10) / 2
// m.next(3) = (1 + 10 + 3) / 3
// m.next(5) = (10 + 3 + 5) / 3
class MovingAverage {
    Queue<Integer> window;
    int size;
    /** Initialize your data structure here. */
    public MovingAverage(int size) {
        window = new LinkedList<Integer>();
        this.size = size;
    }
    
    public double next(int val) {
        window.offer(val);
        if(window.size() > size)window.poll();
        return compute();
    }
    public double compute(){
        double sum = 0;
        int len = window.size();
        for(int i = 0; i < len; i++){
            int temp = window.poll();
            sum+= temp;
            window.offer(temp);
        }
        return sum/len;
    }
}

/**
 * Your MovingAverage object will be instantiated and called as such:
 * MovingAverage obj = new MovingAverage(size);
 * double param_1 = obj.next(val);
 */


// 42 // 736. Parse Lisp Expression
// You are given a string expression representing a Lisp-like expression to return the integer value of.
// The syntax for these expressions is given as follows.
// An expression is either an integer, a let-expression, an add-expression, a mult-expression, or an assigned variable. Expressions always evaluate to a single integer.
// (An integer could be positive or negative.)
// A let-expression takes the form (let v1 e1 v2 e2 ... vn en expr), where let is always the string "let", then there are 1 or more pairs of alternating variables and expressions, meaning that the first variable v1 is assigned the value of the expression e1, the second variable v2 is assigned the value of the expression e2, and so on sequentially; and then the value of this let-expression is the value of the expression expr.
// An add-expression takes the form (add e1 e2) where add is always the string "add", there are always two expressions e1, e2, and this expression evaluates to the addition of the evaluation of e1 and the evaluation of e2.
// A mult-expression takes the form (mult e1 e2) where mult is always the string "mult", there are always two expressions e1, e2, and this expression evaluates to the multiplication of the evaluation of e1 and the evaluation of e2.
// For the purposes of this question, we will use a smaller subset of variable names. A variable starts with a lowercase letter, then zero or more lowercase letters or digits. Additionally for your convenience, the names "add", "let", or "mult" are protected and will never be used as variable names.
// Finally, there is the concept of scope. When an expression of a variable name is evaluated, within the context of that evaluation, the innermost scope (in terms of parentheses) is checked first for the value of that variable, and then outer scopes are checked sequentially. It is guaranteed that every expression is legal. Please see the examples for more details on scope.

class Solution {
    public int evaluate(String expression) {
        return helper(expression, new HashMap<String, Integer>());
    }
    public int helper(String exp, Map<String, Integer> parent){
        if(exp.charAt(0) != '('){
            if(Character.isDigit(exp.charAt(0)) || exp.charAt(0) == '-'){
                return Integer.parseInt(exp);
            }
            else return parent.get(exp);
        }
        Map<String, Integer> map = new HashMap<>();
        map.putAll(parent);
        List<String> para = split(exp.substring(exp.charAt(1) == 'm'? 6: 5, exp.length()-1));
        if(exp.startsWith("(a")){
            return helper(para.get(0), map) + helper(para.get(1), map);
        }
        else if(exp.startsWith("(m")){
            return helper(para.get(0), map)*helper(para.get(1), map);
        }
        else{
            for(int i = 0; i < para.size()-1; i+=2){
                map.put(para.get(i), helper(para.get(i+1), map));
            }
            return helper(para.get(para.size()-1), map);
        }
    }
    public List<String> split(String exp){
        List<String> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        int count = 0;
        for(char c : exp.toCharArray()){
            if(c == '(') count++;
            if(c == ')') count--;
            if(count == 0 && c == ' '){
                res.add(new String(sb.toString()));
                sb = new StringBuilder();
            }
            else{
                sb.append(c);
            }
        }
        if(sb.length() > 0){
            res.add(new String(sb.toString()));
        }
        return res;
    }
}

// 43 // 56. Merge Intervals
// Given a collection of intervals, merge all overlapping intervals.
// Example 1:
// Input: [[1,3],[2,6],[8,10],[15,18]]
// Output: [[1,6],[8,10],[15,18]]
// Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
class Solution {
    public List<Interval> merge(List<Interval> intervals) {
        List<Interval> res = new ArrayList<>();
        if(intervals == null || intervals.size() == 0) return res;
        Comparator<Interval> intervalCmp = new Comparator<Interval>(){
            public int compare(Interval a, Interval b){
                if(a.start == b.start) return a.end - b.end;
                return a.start - b.start;
            }
        };
        PriorityQueue<Interval> queue = new PriorityQueue<>(intervals.size(), intervalCmp);
        for(Interval x : intervals){
            queue.offer(x);
        }
        Interval pre = queue.poll();
        while(!queue.isEmpty()){
            Interval temp = queue.poll();
            if(pre.end >= temp.start){
                pre = fushion(pre, temp);
            }
            else{
                res.add(pre);
                pre = temp;
            }
        }
        res.add(pre);
        return res;
    }
    public Interval fushion(Interval a, Interval b){
        return new Interval(a.start, Math.max(a.end, b.end));
    }
}

// 44 // 723. Candy Crush
// This question is about implementing a basic elimination algorithm for Candy Crush.
// Given a 2D integer array board representing the grid of candy, different positive integers board[i][j] represent different types of candies. A value of board[i][j] = 0 represents that the cell at position (i, j) is empty. The given board represents the state of the game following the player's move. Now, you need to restore the board to a stable state by crushing candies according to the following rules:
// If three or more candies of the same type are adjacent vertically or horizontally, "crush" them all at the same time - these positions become empty.
// After crushing all candies simultaneously, if an empty space on the board has candies on top of itself, then these candies will drop until they hit a candy or bottom at the same time. (No new candies will drop outside the top boundary.)
// After the above steps, there may exist more candies that can be crushed. If so, you need to repeat the above steps.
// If there does not exist more candies that can be crushed (ie. the board is stable), then return the current board.
// You need to perform the above rules until the board becomes stable, then return the current board.
class Solution {
    public int[][] candyCrush(int[][] board) {
        Set<Coordinates> set = new HashSet<>();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                int cur = board[i][j];
                if (cur == 0) continue;
                if ((i - 2 >= 0 && board[i - 1][j] == cur && board[i - 2][j] == cur) ||                                                 // check left 2 candies
                   (i + 2 <= board.length - 1 && board[i + 1][j] == cur && board[i + 2][j] == cur) ||                                   // check right 2 candies
                   (j - 2 >= 0 && board[i][j - 1] == cur && board[i][j - 2] == cur) ||                                                 // check 2 candies top
                   (j + 2 <= board[i].length - 1 && board[i][j + 1] == cur && board[i][j + 2] == cur) ||                               // check 2 candies below
                   (i - 1 >= 0 && i + 1 <= board.length - 1 && board[i - 1][j] == cur && board[i + 1][j] == cur) ||                    // check if it is a mid candy in row
                   (j - 1 >= 0 && j + 1 <= board[i].length - 1 && board[i][j - 1] == cur && board[i][j + 1] == cur)) {                // check if it is a mid candy in column
                    set.add(new Coordinates(i, j));
                }
            }
        }
        if (set.isEmpty()) return board;      //stable board
        for (Coordinates c : set) {
            int x = c.i;
            int y = c.j;
            board[x][y] = 0;      // change to "0"
        }
        drop(board);
        return candyCrush(board);
    }
    private void drop(int[][] board) {                                          // using 2-pointer to "drop"
        for (int j = 0; j < board[0].length; j++) {
            int bot = board.length - 1;
            int top = board.length - 1;
            while (top >= 0) {
                if (board[top][j] == 0) {
                    top--;
                }
                else {
                    board[bot--][j] = board[top--][j];
                }
            }
            while (bot >= 0) {
                board[bot--][j] = 0;
            }
        }
    }
}

class Coordinates {
    int i;
    int j;
    Coordinates(int x, int y) {
        i = x;
        j = y;
    }
}

// 45 // 734. Sentence Similarity
// Given two sentences words1, words2 (each represented as an array of strings), and a list of similar word pairs pairs, determine if two sentences are similar.
// For example, "great acting skills" and "fine drama talent" are similar, if the similar word pairs are pairs = [["great", "fine"], ["acting","drama"], ["skills","talent"]].
// Note that the similarity relation is not transitive. For example, if "great" and "fine" are similar, and "fine" and "good" are similar, "great" and "good" are not necessarily similar.
// However, similarity is symmetric. For example, "great" and "fine" being similar is the same as "fine" and "great" being similar.
// Also, a word is always similar with itself. For example, the sentences words1 = ["great"], words2 = ["great"], pairs = [] are similar, even though there are no specified similar word pairs.
// Finally, sentences can only be similar if they have the same number of words. So a sentence like words1 = ["great"] can never be similar to words2 = ["doubleplus","good"].
class Solution {
    public boolean areSentencesSimilar(String[] words1, String[] words2, String[][] pairs) {
        Map<String, List<String>> map = new HashMap<>();
        for(String[] pair : pairs){
            if(map.containsKey(pair[0])){
                List<String> list = map.get(pair[0]);
                list.add(pair[1]);
                map.put(pair[0], list);
            }
            else{
                List<String> list = new ArrayList<>();
                list.add(pair[1]);
                map.put(pair[0], list);
            }
            if(map.containsKey(pair[1])){
                List<String> list2 = map.get(pair[1]);
                list2.add(pair[0]);
                map.put(pair[1], list2);
            }
            else{
                List<String> list2 = new ArrayList<>();
                list2.add(pair[0]);
                map.put(pair[1], list2);
            }
        }
        if(words1.length != words2.length) return false;
        for(int i = 0; i < words1.length; i++){
            if(words1[i].equals(words2[i])) continue;
            if(map.containsKey(words1[i])){
                boolean flag = false;
                for(String sim : map.get(words1[i])){
                    //System.out.println(sim);
                   if(words2[i].equals(sim)){
                      flag = true;
                       break;
                   }
                }
                if(!flag)
                 return false;
            }
            else return false;
        }
        return true;
    }
}

// 46 // 685. Redundant Connection II
// In this problem, a rooted tree is a directed graph such that, there is exactly one node (the root) for which all other nodes are descendants of this node, plus every node has exactly one parent, except for the root node which has no parents.
// The given input is a directed graph that started as a rooted tree with N nodes (with distinct values 1, 2, ..., N), with one additional directed edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.
// The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [u, v] that represents a directed edge connecting nodes u and v, where u is a parent of child v.
// Return an edge that can be removed so that the resulting graph is a rooted tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array.
class Solution {
    public int[] findRedundantDirectedConnection(int[][] edges) {
        if(edges == null || edges.length == 0) return new int[2];
        int[] can1 = new int[]{-1, -1};
        int[] can2 = new int[]{-1, -1};
        int[] father = new int[edges.length*2+1];
        for(int[] edge : edges){
            if(father[edge[1]] == 0){
                father[edge[1]] = edge[0];
            }
            else{
                can1 = new int[]{father[edge[1]], edge[1]};
                can2 = new int[]{edge[0], edge[1]};
                edge[1] = 0;
            }
        }
        for(int i = 0; i < father.length; i++){
            father[i] = i;
        }
        for(int[] e : edges){
            if(e[1] == 0) continue;
            int p = e[0];
            int c = e[1];
            int pf = find(p, father);
            if(pf == c){
                if(can1[0] == -1) return e;
                else{
                    return can1;
                }
            }
            father[c] = p;
        }
        return can2;
    }
    public int find(int p, int[] father){
        if(father[p] == p) return p;
        return find(father[p], father);
    }
}

// 47 // 228. Summary Ranges
// Given a sorted integer array without duplicates, return the summary of its ranges.

// Example 1:

// Input:  [0,1,2,4,5,7]
// Output: ["0->2","4->5","7"]
// Explanation: 0,1,2 form a continuous range; 4,5 form a continuous range.
// Example 2:

// Input:  [0,2,3,4,6,8,9]
// Output: ["0","2->4","6","8->9"]
// Explanation: 2,3,4 form a continuous range; 8,9 form a continuous range.

/**
 * For each starting number S, find its most right element E which satisfies E - S == indexE - indexS.
 * @author xuechao
 *
 */    
public List<String> summaryRanges(int[] nums) {
	List<String> ret = new ArrayList<>();
	//edge cases
	if (nums == null || nums.length == 0) {
		return ret;
	}
	//binary search. Finding most right element satisfying
	int index = 0;
	while (index < nums.length) {
		int start = nums[index];
		int lo = index;
		int hi = nums.length -1 ;
		while (lo < hi) {
			int mid = lo + (hi - lo)/2;
			if (nums[mid] > start + mid - index) {
				//get rid of unsatisfying ones
				hi = mid - 1;
			} else {
				//keep possible candidate
				lo = mid;
				//tie breaking. 
				if (lo == hi - 1) {
					if (nums[hi] == nums[lo] + 1) lo++;
					break;
				}
			}
		}
		
		if (lo == index) {
			ret.add("" + start);
		} else {
			ret.add(start + "->" + nums[lo]);
		}
            index =  lo + 1;
	}
		
	return ret;
}

// 48 // 375. Guess Number Higher or Lower II
// We are playing the Guess Game. The game is as follows:
// I pick a number from 1 to n. You have to guess which number I picked.
// Every time you guess wrong, I'll tell you whether the number I picked is higher or lower.
// However, when you guess a particular number x, and you guess wrong, you pay $x. You win the game when you guess the number I picked.
// Example:
// n = 10, I pick 8.
// First round:  You guess 5, I tell you that it's higher. You pay $5.
// Second round: You guess 7, I tell you that it's higher. You pay $7.
// Third round:  You guess 9, I tell you that it's lower. You pay $9.
// Game over. 8 is the number I picked.
// You end up paying $5 + $7 + $9 = $21.
class Solution {
public:
    int getMoneyAmount(int n) {
        vector<vector<int>> maz(n+1,vector<int>(n+1,0));
        return solve(maz,1,n);
    }
    int solve(vector<vector<int>>& maz,int b, int t)
    {
        if(b>=t)return 0;
        if(maz[b][t])return maz[b][t];
        maz[b][t]=0xfffffff;
        for(int i=b;i<=t;i++)
        {
            maz[b][t]=min(maz[b][t],i+max(solve(maz,b,i-1),solve(maz,i+1,t)));
        }
        return maz[b][t];
    }
};

// 49 // 380. Insert Delete GetRandom O(1)
// Design a data structure that supports all following operations in average O(1) time.
// insert(val): Inserts an item val to the set if not already present.
// remove(val): Removes an item val from the set if present.
// getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.
class RandomizedSet {
public:
    /** Initialize your data structure here. */
    RandomizedSet() {}
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    bool insert(int val) {
        if (m.count(val)) return false;
        nums.push_back(val);
        m[val] = nums.size() - 1;
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    bool remove(int val) {
        if (!m.count(val)) return false;
        int last = nums.back();
        m[last] = m[val];
        nums[m[val]] = last;
        nums.pop_back();
        m.erase(val);
        return true;
    }
    
    /** Get a random element from the set. */
    int getRandom() {
        return nums[rand() % nums.size()];
    }
private:
    vector<int> nums;
    unordered_map<int, int> m;
};

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet obj = new RandomizedSet();
 * bool param_1 = obj.insert(val);
 * bool param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */


// 50 // 158. Read N Characters Given Read4 II - Call multiple times
// The API: int read4(char *buf) reads 4 characters at a time from a file.
// The return value is the actual number of characters read. For example, it returns 3 if there is only 3 characters left in the file.
// By using the read4 API, implement the function int read(char *buf, int n) that reads n characters from the file.
// Note:
// The read function may be called multiple times.
/* The read4 API is defined in the parent class Reader4.
      int read4(char[] buf); */

public class Solution extends Reader4 {
    /**
     * @param buf Destination buffer
     * @param n   Maximum number of characters to read
     * @return    The number of characters read
     */
     private int buffPtr = 0;
    private int buffCnt = 0;
    private char[] buff = new char[4];
    public int read(char[] buf, int n) {
        int ptr = 0;
        while (ptr < n) {
            if (buffPtr == 0) {
                buffCnt = read4(buff);
            }
            if (buffCnt == 0) break;
            while (ptr < n && buffPtr < buffCnt) {
                buf[ptr++] = buff[buffPtr++];
            }
            if (buffPtr == buffCnt) buffPtr = 0;
        }
        return ptr;
    }
}
