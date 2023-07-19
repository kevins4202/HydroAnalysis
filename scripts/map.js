google.charts.load('current', {
    'packages':['geochart'],
  });
  google.charts.setOnLoadCallback(drawRegionsMap);

  function drawRegionsMap() {
    var data = google.visualization.arrayToDataTable([
      ['Country', 'Publications'],
      ['China', 10867],
['United States', 5488],
['Australia', 1706],
['Canada', 1413],
['India', 1394],
['Germany', 1167],
['United Kingdom', 1084],
['Iran', 971],
['France', 924],
['Italy', 922],
['Netherlands', 902],
['Spain', 900],
['South Korea', 829],
['Brazil', 628],
['Japan', 609],
['Switzerland', 442]
    ]);

    var options = {};

    var chart = new google.visualization.GeoChart(document.getElementById('regions_div'));

    chart.draw(data, options);
  }