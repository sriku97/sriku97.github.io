var gameArray = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
var score = 0;

function cat(a,b) //function to concatenate two numbers
{
	return 10*a+b;
}
function getRandom()//generate random integer from 0-3
{
    return (Math.floor(Math.random() * (4)) );
}
function randomGen()//function to generate the position of the new tile
{
	var check = 1;
	var pos = [];
	do
	{
		pos.pop();
		pos.pop();
		pos.push(getRandom());
		pos.push(getRandom());
		if(gameArray[pos[0]][pos[1]]==0)
		{
			check = -1;
		}
	}while(check>0);
	return pos;
}
function checkLeft(i,j)
{
	if(j==0)
		return 0;
	else if((gameArray[i][j-1]>0)&&(gameArray[i][j-1]==gameArray[i][j]))
		return 2;
	else if((gameArray[i][j-1]>0)&&(gameArray[i][j-1]!=gameArray[i][j]))
		return 0;
	else return 1;
}
function checkUp(i,j)
{
	if(i==0)
		return 0;
	else if((gameArray[i-1][j]>0)&&(gameArray[i-1][j]==gameArray[i][j]))
		return 2;
	else if((gameArray[i-1][j]>0)&&(gameArray[i-1][j]!=gameArray[i][j]))
		return 0;
	else return 1;
}
function checkDown(i,j)
{
	if(i==3)
		return 0;
	else if((gameArray[i+1][j]>0)&&(gameArray[i+1][j]==gameArray[i][j]))
		return 2;
	else if((gameArray[i+1][j]>0)&&(gameArray[i+1][j]!=gameArray[i][j]))
		return 0;
	else return 1;
}
function checkRight(i,j)
{
	if(j==3)
		return 0;
	else if((gameArray[i][j+1]>0)&&(gameArray[i][j+1]==gameArray[i][j]))
		return 2;
	else if((gameArray[i][j+1]>0)&&(gameArray[i][j+1]!=gameArray[i][j]))
		return 0;
	else return 1;
}
function display()
{
	$('.box').empty();
	$('.box').css('background-color','lightgray');
	for(i=0;i<4;i++)
	{
		for(j=0;j<4;j++)
		{
			if(gameArray[i][j]>0)
			{
				boxstring = ".box"+cat(i,j);
				$(boxstring).text(gameArray[i][j]);
				switch(gameArray[i][j])
				{
					case 2: $(boxstring).css("background-color","white");
							break;
					case 4: $(boxstring).css("background-color","#ffff99");
							break;
					case 8: $(boxstring).css("background-color","#ff9933");
							break;
					case 16: $(boxstring).css("background-color","#ff6600");
							break;
					case 32: $(boxstring).css("background-color","#ff3300");
							break;
					case 64: $(boxstring).css("background-color","#ff0000");
							break;
					default: $(boxstring).css("background-color","#ffcc00");
							break;
				}
			}
		}
	}
	$('#score').empty();
	$("#score").text("Score: "+score);
}
function animate(box)
{
	$(box).css('opacity','0.5')
	$(box).animate({opacity:'1'},"fast");
}
var addtile = function()
{
	var i,j;
	var count = 1;
	for(i=0;i<4;i++)
	{
		for(j=0;j<4;j++)
		{
			if(gameArray[i][j]>0)
				count++;
		}
	}
	if(count>=16)
	{
		const finalscore = score;
		$(".game").hide();
		$(".gameover").show();
		$(".gameover").text("Game Over!");
		$("#score").hide();
		$("#finalscore").show();
		$("#finalscore").text("Score: "+finalscore);
	}
	var newbox = randomGen();
	var newboxstring = ".box"+cat(newbox[0],newbox[1]);
	animate(newboxstring);
	gameArray[newbox[0]][newbox[1]] = 2;
	display();
	//console.log(gameArray);
};
var game = function(e)
{
	var i,j,temp;
	var check = 0;
	switch(e.keyCode)
	{
		//when left is pressed
		case 37: for(j=0;j<4;j++)
				  {
				  	  for(i=0;i<4;i++)
				  	  {
				  	      if(gameArray[i][j]>0)
				  	      {
				  	      	  temp = j;
				  	      	  while(checkLeft(i,temp)>0)
				  	      	  {
				  	      	  	  if(checkLeft(i,temp)==1)
				  	      	  	  {
				  	      	  	  	  gameArray[i][temp-1] = gameArray[i][temp];
				  	      	  	  	  gameArray[i][temp] = 0;
				  	      	  	  	  --temp;
				  	      	  	  	  check = 1;
				  	      	  	  }
				  	      	  	  if(checkLeft(i,temp)==2)
				  	      	  	  {
				  	      	  	  	  gameArray[i][temp-1] += gameArray[i][temp];
				  	      	  	  	  gameArray[i][temp] = 0;
				  	      	  	  	  animate(".box"+cat(i,temp-1));
				  	      	  	  	  score += gameArray[i][temp-1];
				  	      	  	  	  --temp;
				  	      	  	  	  check = 1;
				  	      	  	  	  break;
				  	      	  	  }
				  	      	  }
				  	      }
				  	  }
				  }
				  break;
		//when up is pressed
		case 38: for(i=0;i<4;i++)
				  {
				  	  for(j=0;j<4;j++)
				  	  {
				  	      if(gameArray[i][j]>0)
				  	      {
				  	      	  temp = i;
				  	      	  while(checkUp(temp,j)>0)
				  	      	  {
				  	      	  	  if(checkUp(temp,j)==1)
				  	      	  	  {
				  	      	  	  	  gameArray[temp-1][j] = gameArray[temp][j];
				  	      	  	  	  gameArray[temp][j] = 0;
				  	      	  	  	  --temp;
				  	      	  	  	  check = 1;
				  	      	  	  }
				  	      	  	  if(checkUp(temp,j)==2)
				  	      	  	  {
				  	      	  	  	  gameArray[temp-1][j] += gameArray[temp][j];
				  	      	  	  	  gameArray[temp][j] = 0;
				  	      	  	  	  animate(".box"+cat(temp-1,j));
				  	      	  	  	  score += gameArray[temp-1][j];
				  	      	  	  	  --temp;
				  	      	  	  	  check = 1;
				  	      	  	  	  break;
				  	      	  	  }
				  	      	  }
				  	      }
				  	  }
				  }
				  break;
		//when down is pressed
		case 40: for(i=3;i>=0;i--)
				  {
				  	  for(j=0;j<4;j++)
				  	  {
				  	      if(gameArray[i][j]>0)
				  	      {
				  	      	  temp = i;
				  	      	  while(checkDown(temp,j)>0)
				  	      	  {
				  	      	  	  if(checkDown(temp,j)==1)
				  	      	  	  {
				  	      	  	  	  gameArray[temp+1][j] = gameArray[temp][j];
				  	      	  	  	  gameArray[temp][j] = 0;
				  	      	  	  	  ++temp;
				  	      	  	  	  check = 1;
				  	      	  	  }
				  	      	  	  if(checkDown(temp,j)==2)
				  	      	  	  {
				  	      	  	  	  gameArray[temp+1][j] += gameArray[temp][j];
				  	      	  	  	  gameArray[temp][j] = 0;
				  	      	  	  	  animate(".box"+cat(temp+1,j));
				  	      	  	  	  score += gameArray[temp+1][j];
				  	      	  	  	  ++temp;
				  	      	  	  	  check = 1;
				  	      	  	  	  break;
				  	      	  	  }
				  	      	  }
				  	      }
				  	  }
				  }
				  break;
		//when right is pressed
		case 39: for(j=3;j>=0;j--)
				  {
				  	  for(i=0;i<4;i++)
				  	  {
				  	      if(gameArray[i][j]>0)
				  	      {
				  	      	  temp = j;
				  	      	  while(checkRight(i,temp)>0)
				  	      	  {
				  	      	  	  if(checkRight(i,temp)==1)
				  	      	  	  {
				  	      	  	  	  gameArray[i][temp+1] = gameArray[i][temp];
				  	      	  	  	  gameArray[i][temp] = 0;
				  	      	  	  	  ++temp;
				  	      	  	  	  check = 1;
				  	      	  	  }
				  	      	  	  if(checkRight(i,temp)==2)
				  	      	  	  {
				  	      	  	  	  gameArray[i][temp+1] += gameArray[i][temp];
				  	      	  	  	  gameArray[i][temp] = 0;
				  	      	  	  	  animate(".box"+cat(i,temp+1));
				  	      	  	  	  score += gameArray[i][temp+1];
				  	      	  	  	  ++temp;
				  	      	  	  	  check = 1;
				  	      	  	  	  break;
				  	      	  	  }
				  	      	  }
				  	      }
				  	  }
				  }
				  break;
		default:  return;
	};
	if(check == 1)
		addtile();
};
$(document).ready(function(){
	//generate background
	$(".gameover").hide();
	$("#finalscore").hide();
	$("#score").text("Score: "+score);
	for(i=0;i<4;i++)
	{
		for(j=0;j<4;j++)
		{
			var box = $("<div/>").addClass('box').addClass('box'.concat(cat(i,j)));
			switch(cat(i,j))
			{
				case 00: box.addClass('box-topleft');
						break;
				case 01:
				case 02: box.addClass('box-top');
						break;
				case 03: box.addClass('box-topright');
						break;
				case 10:
				case 20: box.addClass('box-left');
						break;
				case 11:
				case 12:
				case 21:
				case 22: box.addClass('box-middle');
						 break;
				case 13:
				case 23: box.addClass('box-right');
						 break;
				case 30: box.addClass('box-bottomleft');
						 break;
				case 31:
				case 32: box.addClass('box-bottom');
						 break;
				case 33: box.addClass('box-bottomright');
						 break;
			}
			$(".game").append(box);
		}
	}
	addtile();
	addtile();
	document.addEventListener("keydown",function(){ game(event);});
});