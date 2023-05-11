function calculate_max_width() {
    const window_width = $(window).width();
    const max_width = Math.floor(window_width * 0.8); // Adjust this factor according to your layout and design
    return max_width;
}


$(document).ready(function () {
    $("#generate-lyrics-form").submit(function (event) {
        event.preventDefault();
        const artist = $("#artist").val();
        const prompt = $("#prompt").val();

        //Show the loading indicator
        $("#loading").show();
        // Hide the previously generated lyrics
        $("#generated-lyrics").html("");

        $.post("/generate_lyrics", {artist: artist, prompt: prompt, max_width: calculate_max_width() }, function (data) {
            // Hide the loading indicator
            $("#loading").hide();
            // Display the lyrics
            $("#generated-lyrics").html(`<pre>${data.lyrics}</pre>`);
        });
    });
});
