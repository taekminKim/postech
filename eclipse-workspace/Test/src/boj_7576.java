import java.io.IOException;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

public class boj_7576 {
	static int arr[][];
	static boolean visit[][];
	static int dx[] = {-1,0,1,0};
	static int dy[] = {0,1,0,-1};
	static int N, M;
	static int cnt;
	public static void main(String[] args) throws IOException {
		Scanner sc = new Scanner(System.in);
		N = sc.nextInt();
		M = sc.nextInt();
		arr = new int[N][M];
		visit = new boolean[N][M];
		cnt = 0;
		Queue<Point> q = new LinkedList<Point>();
		for(int i=0; i<N; i++) {
			for(int j=0; j<M; j++) {
				arr[i][j]= sc.nextInt();
				if(arr[i][j]==1)
					q.add(new Point(i,j));
			}
		}
		
		while(!q.isEmpty()) {
			Point temp_p = q.poll();
			visit[temp_p.x][temp_p.y] = true; 
			for(int i=0; i<4; i++) {
				
				int nx = temp_p.x + dx[i];
				int ny = temp_p.y + dy[i];
				
				
			}
			
			
		}
		
		
	}
}
