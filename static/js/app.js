$(document).ready(function () {
    $("#generate-lyrics-form").submit(function (event) {
        event.preventDefault();
        const artist = $("#artist").val();
        const prompt = $("#prompt").val();

        //Show the loading indicator
        $("#loading").show();
        // Hide the previously generated lyrics
        $("#generated-lyrics").html("");

        $.post("/generate_lyrics", {artist: artist, prompt: prompt }, function (data) {
            // Hide the loading indicator
            $("#loading").hide();
            // Display the lyrics
            $("#generated-lyrics").html(`<div style="white-space: pre-line; word-wrap: break-word; max-width: 80%">${data.lyrics}</div>`);
        });
    });
});
