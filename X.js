let y = 0
let z = 2

function sonuc_bul(x){
    for(let i =1; i<=z; i++)
    {
        y = y+((20+x*i-4)/2)
    }
    return y
}


console.log(sonuc_bul(sonuc_bul(4)));