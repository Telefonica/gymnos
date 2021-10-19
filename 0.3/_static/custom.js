/*
*
*   Custom js for sphinx pages
*
*/


function splitWithTail(str,delim,count){
    const parts = str.split(delim)
    const tail = parts.slice(count).join(delim)
    const result = parts.slice(0,count)
    result.push(tail)
    return result
}

function buildTable(labels, objects) {
    const table = document.createElement('table')
    const thead = document.createElement('thead')
    const tbody = document.createElement('tbody')

    table.className = "docutils align-default"

    const theadTr = document.createElement('tr')
    for (let i = 0; i < labels.length; i++) {
        const theadTh = document.createElement('th')
        theadTh.className = "head"
        theadTh.innerHTML = labels[i]
        theadTr.appendChild(theadTh)
    }
    thead.appendChild(theadTr)
    table.appendChild(thead)

    for (let j = 0; j < objects.length; j++) {
        const tbodyTr = document.createElement('tr')
        for (let k = 0; k < labels.length; k++) {
            const tbodyTd = document.createElement('td')
            tbodyTd.innerHTML = objects[j][labels[k].toLowerCase()]
            tbodyTr.appendChild(tbodyTd)
        }
        tbody.appendChild(tbodyTr)
    }
    table.appendChild(tbody)

    return table
}


function editHydraParamsTable() {
    const container = document.querySelector(".gymnos-hydra")

    if (!container) {
        return
    }

    const params = {}

    Array.from(document.querySelectorAll(".gymnos-hydra .sig-param")).forEach(elem => {
        const paramElems = elem.querySelectorAll(".n")

        const paramNameElem = paramElems[0]
        const paramTypeElem = paramElems[1]
        const defaultValueElem = elem.querySelector(".default_value")

        if (!paramNameElem) {
            return
        }

        const paramName = paramNameElem.textContent

        params[paramName] = {
            "default": null,
            "type": null,
            "description": null
        }

        if (defaultValueElem) {
            let defaultValue = defaultValueElem.textContent

            if (defaultValue === "None") {
                defaultValue = "null"
            }

            params[paramName]["default"] = `<code class="docutils literal notranslate"><span class="pre">${defaultValue}</span></code>`
        }

        if (paramTypeElem) {
            params[paramName]["type"] = paramTypeElem.textContent
        }
    })

    Array.from(document.querySelectorAll(".gymnos-hydra .field-list li p")).forEach(elem => {
        const paramNameElem = elem.querySelector("strong")

        if (!paramNameElem) {
            return
        }

        const paramName = paramNameElem.textContent

        const [title, description] = splitWithTail(elem.innerHTML, " – ", 2)

        params[paramName]["description"] = description
    })

    const tableObjects = []

    for (let [key, value] of Object.entries(params)) {
        tableObjects.push({
            "parameter": key,
            "default": value["default"],
            "description": value["description"],
            "required?": value["default"] ? null : "<p style='text-align: center'>&#11093;</p>"
        })
    }

    const tableElem = buildTable(["Parameter", "Required?", "Default", "Description"], tableObjects)

    container.prepend(tableElem)
}


document.addEventListener("DOMContentLoaded", editHydraParamsTable)
