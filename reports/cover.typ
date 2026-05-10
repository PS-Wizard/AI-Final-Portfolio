#let cover-page(report-title) = block(width: 100%)[
  #grid(
    columns: (1fr, 1fr),
    align: (left, right),
    image("assets/uow-logo.png", width: 46%),
    image("assets/herald-ing-logo.png", width: 46%),
  )

  #v(4.2em)

  #align(center)[
    #text(size: 15pt, tracking: 0.8pt)[6CS012]
    #v(1.0em)
    #text(size: 22pt)[Artificial Intelligence and]
    #v(0.35em)
    #text(size: 22pt)[Machine Learning]
    #v(1.4em)
    #line(length: 34%, stroke: 0.45pt + luma(150))
    #v(1.25em)
    #text(size: 15pt)[#report-title]
  ]

  #v(5em)

  #align(center)[
    #block(width: 58%)[
      #set text(size: 11pt)
      #grid(
        columns: (1.2fr, 2.4fr),
        row-gutter: 0.85em,
        column-gutter: 1.1em,
        align: (left, left),
        [Name], [Swoyam Pokharel],
        [Group], [15G],
        [Tutor], [Jinu Nyachhyon],
        [Module Leader], [Siman Giri],
      )
    ]
  ]
]
