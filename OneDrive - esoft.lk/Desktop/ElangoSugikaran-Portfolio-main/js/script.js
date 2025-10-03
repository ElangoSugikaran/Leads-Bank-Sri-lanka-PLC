// "learn more button in about me"
/*
document.addEventListener("DOMContentLoaded", function() {
    const toggleButton = document.getElementById("toggleInfo");
    const additionalInfo = document.getElementById("additionalInfo");

    toggleButton.addEventListener("click", function() {
        if (additionalInfo.style.display === "none") {
            additionalInfo.style.display = "block";
        } else {
            additionalInfo.style.display = "none";
        }
    });
});
*/

document.addEventListener("DOMContentLoaded", function() {
    const toggleButton = document.getElementById("toggleInfo");
    const additionalInfo = document.getElementById("additionalInfo");

    toggleButton.addEventListener("click", function(event) {
        event.preventDefault(); // Prevent default behavior of the anchor tag
        if (additionalInfo.style.display === "none") {
            additionalInfo.style.display = "block";
        } else {
            additionalInfo.style.display = "none";
        }
    });
});
