// 681. Next Closest Time
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

// 683. K Empty Slots
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

// 159. Longest Substring with At Most Two Distinct Characters
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

// 399. Evaluate Division
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

// 843. Guess the Word
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

// 857. Minimum Cost to Hire K Workers
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

// 844. Backspace String Compare
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

// 904. Fruit Into Baskets
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

// 489. Robot Room Cleaner
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

// 803. Bricks Falling When Hit
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

// 686. Repeated String Match
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

//299. Bulls and Cows
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

//562. Longest Line of Consecutive One in Matrix （这道题没做，贴的是discussion里面点赞高的答案）
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

//135. Candy
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

//308. Range Sum Query 2D - Mutable
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

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix obj = new NumMatrix(matrix);
 * obj.update(row,col,val);
 * int param_2 = obj.sumRegion(row1,col1,row2,col2);
 */






