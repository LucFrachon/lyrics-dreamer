$(document).ready(function () {
    $("#lyrics-generator-form").submit(function (event) {
        event.preventDefault();
        const artist = $("#artist").val();
        const prompt = $("#prompt").val();

        //Show the loading indicator
        $("#loading").show();
        // Hide the previously generated lyrics
        $("#generated-lyrics").html("");

        $.post("/lyrics_generator", {artist: artist, prompt: prompt }, function (data) {
            // Hide the loading indicator
            $("#loading").hide();
            // Display the lyrics
            $("#generated-lyrics").html(
                `<div style="white-space: pre-line; word-wrap: break-word; max-width: 80%">${data.lyrics}</div>`
            );
        });
    });

    // Event listener for generating lyrics and starting the game
    $("#generate-game-lyrics-btn").click(function () {
        $.post("/generate_game_lyrics", function (data) {
            $("#lyrics-display").html(
                `<div style="white-space: pre-line; word-wrap: break-word; max-width: 80%">${data.lyrics}</div>`
            );
            $("#result-display").text(""); // Clear the result display
            enableGuessForm(); // Enable the guess form
        });
    });

    // Event listener for submitting a guess
    $("#guess-form").submit(function (event) {
        event.preventDefault();
        const userGuess = $("#artist-select").val(); // Get the selected artist from the dropdown
        $.post("/submit_guess", {guess: userGuess}, function (data) {
            // Update the game state based on the returned data
            // For example, data might be a JSON object that includes the result and the updated score
            $("#result-display").text(data.result);
            $("#score-display").text("Score: " + data.score + "%");
            $("#rounds-display").text("Rounds played: " + data.rounds);
            disableGuessForm(); // Disable the guess form after submitting
        });
    });

    // Function to enable the guess form
    function enableGuessForm() {
        $("#artist-select").prop("disabled", false);
        $("#guess-form button").prop("disabled", false);
    }

    // Function to disable the guess form
    function disableGuessForm() {
        $("#artist-select").prop("disabled", true);
        $("#guess-form button").prop("disabled", true);
    }

});
