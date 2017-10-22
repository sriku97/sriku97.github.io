var a = 1;
$( document ).ready(function() {
    $("#myModal").modal();
    $("#myModal").on('hidden.bs.modal', function () {
  		$(this).removeData('bs.modal');
        $('#a'+a).popover('toggle');
		
		$('.next').click(function(){
			$('#a'+a).popover('toggle');
			a=a+1;
			if(a==21)
			{
				a=1;
			}
			$('#a'+a).popover('toggle');
		});

		$('.prev').click(function(){
			$('#a'+a).popover('toggle');
			a=a-1;
			if(a==0)
			{
				a=20;
			}
			$('#a'+a).popover('toggle');
		});


    });
});