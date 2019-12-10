function submit() {
    loading = true;
    let song = document.getElementsByName("song_name")[0].value;
    let artist = document.getElementsByName("artist_name")[0].value;
    searchForSong(song, artist)
    .then(response => {
        listOfSongs(response.message.body.track_list);
        loading = false;
    });
}

function listOfSongs(songs_list) {
    let songs_div = document.querySelector('#songs_list');
    let h2 = document.createElement('h2');
    h2.innerHTML = "Select the correct song:";
    h2.setAttribute('id', 'old_h2');
    let ul = document.createElement('ul');
    ul.setAttribute('id','songs_list');
    ul.setAttribute('class', 'list-group');
    songs_div.appendChild(h2);
    h2.appendChild(ul);

    _.forEach(songs_list, (song) => {
        var button = document.createElement('button');
        button.setAttribute('type', 'button');
        button.setAttribute('class', 'btn');
        button.addEventListener('click', () => {
            getLyrics(song.track.track_id);
        });

        button.innerHTML = button.innerHTML + song.track.track_name;
        var li = document.createElement('li');
        li.setAttribute('class', 'list-group-item');
        li.appendChild(button);
        ul.appendChild(li);
    });
}

function getLyrics(track_id) {
    return new Promise(resolve => {
        fetch(`http://127.0.0.1:8000/get_lyrics/${track_id}/`, {
            method: 'GET', 
            body: null, 
            credentials: "same-origin",
            headers: {
                "Accept": "application/json",
                "Content-Type": "application/json", 
            }
        }).then((raw_response) => raw_response.json())
        .then(data => {
            let mood = data[0] > 0.5 ? 'Happy' : 'Sad';
            let songs_div = document.querySelector('#songs_list');
            let old_h2 = document.querySelector('#old_h2');
            let h2 = document.createElement('h2');
            h2.innerHTML = "Mood of the song is: " + mood + "!";
            songs_div.insertBefore(h2, old_h2);
            songs_div.removeChild(old_h2);
            return resolve(data);
        });
    });

}

function searchForSong(song, artist) {
    return new Promise(resolve => {
        fetch(`http://127.0.0.1:8000/get_song/${song}/${artist}`, {
            method: 'GET', 
            body: null, 
            credentials: "same-origin",
            headers: {
                "Accept": "application/json",
                "Content-Type": "application/json", 
            }
        }).then((raw_response) => raw_response.json())
        .then(data => {
            return resolve(data);
        });
    });
}